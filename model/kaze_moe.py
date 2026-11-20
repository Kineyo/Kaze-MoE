import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass


@dataclass
class KazeConfig:
    vocab_size: int = 32768
    d_model: int = 768
    n_layers: int = 20
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ff: int = 2816
    n_routed_experts: int = 7
    n_shared_experts: int = 1
    top_k: int = 2
    max_seq_len: int = 4096
    rope_theta: float = 100000.0
    bias_update_speed: float = 0.001
    use_checkpoint: bool = False

    @classmethod
    def kaze_2_4b(cls, use_checkpoint: bool = True) -> "KazeConfig":
        """Preset Kaze-2.4B-A530M (~2.41B totales | ~528M activos/token)."""
        return cls(
            vocab_size=32768,
            d_model=1024,
            n_layers=23,
            n_heads=16,
            n_kv_heads=4,
            d_ff=2048,
            n_routed_experts=15,
            n_shared_experts=1,
            top_k=2,
            max_seq_len=4096,
            rope_theta=100000.0,
            bias_update_speed=0.001,
            use_checkpoint=use_checkpoint,
        )

    @classmethod
    def kaze_4_7b(cls, use_checkpoint: bool = False) -> "KazeConfig":
        """Preset Kaze-4.7B-A530M (Sparse Upcycling: 1 Shared + 31 Routed Top-2, ~4.72B totales | ~528M activos)."""
        return cls(
            vocab_size=32768,
            d_model=1024,
            n_layers=23,
            n_heads=16,
            n_kv_heads=4,
            d_ff=2048,
            n_routed_experts=31,
            n_shared_experts=1,
            top_k=2,
            max_seq_len=4096,
            rope_theta=100000.0,
            bias_update_speed=0.001,
            use_checkpoint=use_checkpoint,
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight.float()).type_as(x)


def precompute_rope_cache(dim: int, end: int, theta: float = 100000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)
    cos = freqs_cos.float().unsqueeze(0).unsqueeze(2)
    sin = freqs_sin.float().unsqueeze(0).unsqueeze(2)
    xq_out = torch.stack([xq_r * cos - xq_i * sin, xq_r * sin + xq_i * cos], dim=-1).flatten(3)
    xk_out = torch.stack([xk_r * cos - xk_i * sin, xk_r * sin + xk_i * cos], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class SwiGLUExpert(nn.Module):
    """Experto individual con activación SwiGLU."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class DeepSeekStyleMoE(nn.Module):
    """
    MoE ligero con:
    - 1 Shared Expert siempre activo
    - 7 Routed Experts (Top-2 activo por token)
    - Aux-Loss-Free Load Balancing (actualización diferida por paso, compatible con checkpointing)
    """

    def __init__(self, config: KazeConfig):
        super().__init__()
        self.n_routed = config.n_routed_experts
        self.top_k = config.top_k
        self.bias_update_speed = config.bias_update_speed

        self.shared_expert = SwiGLUExpert(config.d_model, config.d_ff)
        self.experts = nn.ModuleList([
            SwiGLUExpert(config.d_model, config.d_ff) for _ in range(self.n_routed)
        ])
        self.gate = nn.Linear(config.d_model, self.n_routed, bias=False)
        self.register_buffer("expert_bias", torch.zeros(self.n_routed))
        self.register_buffer("_step_counts", torch.zeros(self.n_routed), persistent=False)
        self._step_tokens = 0

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        N = x_flat.shape[0]

        shared_out = self.shared_expert(x_flat)

        raw_logits = self.gate(x_flat)
        routing_scores = raw_logits + self.expert_bias.unsqueeze(0)
        _, top_indices = torch.topk(routing_scores, self.top_k, dim=-1)

        selected_logits = torch.gather(raw_logits, -1, top_indices)
        routing_weights = F.softmax(selected_logits, dim=-1, dtype=torch.float32).to(x.dtype)

        if self.training:
            with torch.no_grad():
                counts = torch.bincount(top_indices.view(-1), minlength=self.n_routed).to(self._step_counts.dtype)
                self._step_counts.add_(counts)
                self._step_tokens += N

        routed_out = torch.zeros_like(x_flat)
        for expert_idx in range(self.n_routed):
            token_mask, k_pos = torch.where(top_indices == expert_idx)
            if token_mask.numel() > 0:
                expert_input = x_flat[token_mask]
                expert_output = self.experts[expert_idx](expert_input)
                weight = routing_weights[token_mask, k_pos].unsqueeze(-1)
                routed_out.index_add_(0, token_mask, expert_output * weight)

        out = (shared_out + routed_out).view(B, T, C)
        z_loss = torch.logsumexp(raw_logits.float(), dim=-1).pow(2).mean()
        return out, z_loss

    @torch.no_grad()
    def update_expert_bias(self):
        if self._step_tokens > 0:
            ideal_count = (self._step_tokens * self.top_k) / self.n_routed
            error = ideal_count - self._step_counts.float()
            self.expert_bias.add_(self.bias_update_speed * torch.sign(error).to(self.expert_bias.dtype))
            self._step_counts.zero_()
            self._step_tokens = 0


class GQAAttention(nn.Module):
    """Grouped-Query Attention con FlashAttention nativa (SDPA)."""

    def __init__(self, config: KazeConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        xq = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        xk = xk.repeat_interleave(self.n_rep, dim=2)
        xv = xv.repeat_interleave(self.n_rep, dim=2)

        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        out = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class TransformerBlock(nn.Module):
    def __init__(self, config: KazeConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = GQAAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.moe = DeepSeekStyleMoE(config)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        x = x + self.attn(self.attn_norm(x), freqs_cos, freqs_sin)
        moe_out, z_loss = self.moe(self.ffn_norm(x))
        x = x + moe_out
        return x, z_loss


class KazeMoEModel(nn.Module):
    def __init__(self, config: KazeConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight

        freqs_cos, freqs_sin = precompute_rope_cache(
            config.d_model // config.n_heads, config.max_seq_len, config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        x = self.tok_embeddings(idx)
        freqs_cos = self.freqs_cos[:T]
        freqs_sin = self.freqs_sin[:T]

        total_z_loss = 0.0
        for layer in self.layers:
            if self.training and self.config.use_checkpoint:
                x, z_loss = checkpoint(layer, x, freqs_cos, freqs_sin, use_reentrant=False)
            else:
                x, z_loss = layer(x, freqs_cos, freqs_sin)
            total_z_loss = total_z_loss + z_loss

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = ce_loss + 1e-4 * (total_z_loss / len(self.layers))

        return logits, loss

    @torch.no_grad()
    def update_expert_biases(self):
        for layer in self.layers:
            layer.moe.update_expert_bias()

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        per_layer_expert = sum(p.numel() for p in self.layers[0].moe.shared_expert.parameters())
        inactive_per_layer = (self.config.n_routed_experts - self.config.top_k) * per_layer_expert
        active = total - (inactive_per_layer * self.config.n_layers)
        return total, active

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.9,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].float() / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cum_probs - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            next_token = torch.gather(sorted_idx, -1, torch.multinomial(sorted_probs, 1))
            idx = torch.cat([idx, next_token], dim=1)
            if eos_id is not None and (next_token == eos_id).all():
                break
        return idx
