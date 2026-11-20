import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class EdgeMoEConfig:
    vocab_size: int = 32768       # Vocabulario compacto (múltiplo de 128)
    d_model: int = 768            # Dimensión oculta
    n_layers: int = 18            # Profundidad del Transformer
    n_heads: int = 12             # Cabezas de Query
    n_kv_heads: int = 4           # Cabezas de Key/Value (GQA 3:1 -> ahorra 66% KV-cache)
    d_ff: int = 1536              # Dimensión intermedia de cada experto SwiGLU
    n_routed_experts: int = 7     # Expertos especializados enrutados
    n_shared_experts: int = 1     # Experto base siempre activo
    top_k: int = 2                # Expertos enrutados activos por token (total = 1 + 2 = 3)
    max_seq_len: int = 4096       # Contexto máximo
    rope_theta: float = 100000.0  # Base RoPE para contexto largo
    bias_update_speed: float = 0.001 # Velocidad de ajuste del Aux-Loss-Free balancing


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 100000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim//2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class SwiGLUExpert(nn.Module):
    """Un experto individual ultra-ligero con activación SwiGLU."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class DeepSeekStyleMoE(nn.Module):
    """
    MoE ligero con:
    - 1 Shared Expert (siempre activo: aprende sintaxis y conocimiento general)
    - 7 Routed Experts (Top-2 activo por token: aprende especialidades)
    - Aux-Loss-Free Load Balancing (evita colapso de expertos sin dañar gradientes)
    """
    def __init__(self, config: EdgeMoEConfig):
        super().__init__()
        self.n_routed = config.n_routed_experts
        self.top_k = config.top_k
        self.bias_update_speed = config.bias_update_speed

        # Experto compartido (siempre encendido)
        self.shared_expert = SwiGLUExpert(config.d_model, config.d_ff)

        # Expertos enrutados
        self.experts = nn.ModuleList([
            SwiGLUExpert(config.d_model, config.d_ff) for _ in range(self.n_routed)
        ])

        # Router lineal
        self.gate = nn.Linear(config.d_model, self.n_routed, bias=False)

        # Sesgo dinámico para balanceo de carga (no recibe gradientes de backprop)
        self.register_buffer("expert_bias", torch.zeros(self.n_routed))

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # (N_tokens, d_model)
        N = x_flat.shape[0]

        # 1. Salida del experto compartido (procesa todos los tokens)
        shared_out = self.shared_expert(x_flat)

        # 2. Cálculo de afinidad del router
        raw_logits = self.gate(x_flat)  # (N, n_routed)
        
        # Sumamos expert_bias SOLO para elegir qué expertos se activan (routing)
        routing_scores = raw_logits + self.expert_bias.unsqueeze(0)
        _, top_indices = torch.topk(routing_scores, self.top_k, dim=-1)  # (N, top_k)

        # Los pesos reales (gates) se calculan sobre los logits limpios sin el bias
        selected_logits = torch.gather(raw_logits, -1, top_indices)
        routing_weights = F.softmax(selected_logits, dim=-1, dtype=torch.float32).to(x.dtype)

        # 3. Actualización del balanceo de carga en modo entrenamiento (Aux-Loss-Free)
        if self.training:
            with torch.no_grad():
                # Contamos cuántos tokens recibió cada experto en este batch
                counts = torch.bincount(top_indices.view(-1), minlength=self.n_routed).float()
                ideal_count = (N * self.top_k) / self.n_routed
                # Si un experto está saturado, bajamos su bias; si está ocioso, lo subimos
                error = ideal_count - counts
                self.expert_bias += self.bias_update_speed * torch.sign(error)

        # 4. Ejecución dispersa (Sparse dispatch) de los expertos elegidos
        routed_out = torch.zeros_like(x_flat)
        for expert_idx in range(self.n_routed):
            # Encontrar qué tokens y en qué posición (0 o 1 del top_k) eligieron este experto
            token_mask, k_pos = torch.where(top_indices == expert_idx)
            if token_mask.numel() > 0:
                expert_input = x_flat[token_mask]
                expert_output = self.experts[expert_idx](expert_input)
                weight = routing_weights[token_mask, k_pos].unsqueeze(-1)
                routed_out.index_add_(0, token_mask, expert_output * weight)

        # Combinar experto compartido + expertos enrutados
        out = (shared_out + routed_out).view(B, T, C)
        
        # Z-loss suave para estabilidad numérica en bfloat16
        z_loss = torch.logsumexp(raw_logits, dim=-1).pow(2).mean()
        return out, z_loss


class GQAAttention(nn.Module):
    """Grouped-Query Attention con FlashAttention nativa (SDPA)."""
    def __init__(self, config: EdgeMoEConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

    def forward(self, x, freqs_cis):
        B, T, _ = x.shape
        xq = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Expandir KV heads para hacer match con Query heads
        xk = xk.repeat_interleave(self.n_rep, dim=2)
        xv = xv.repeat_interleave(self.n_rep, dim=2)

        # Transponer a (B, n_heads, T, head_dim) para FlashAttention
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        out = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class TransformerBlock(nn.Module):
    def __init__(self, config: EdgeMoEConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = GQAAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.moe = DeepSeekStyleMoE(config)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        moe_out, z_loss = self.moe(self.ffn_norm(x))
        x = x + moe_out
        return x, z_loss


class EdgeMoEModel(nn.Module):
    def __init__(self, config: EdgeMoEConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (compartir embeddings de entrada y salida ahorra ~25M de parámetros)
        self.lm_head.weight = self.tok_embeddings.weight

        freqs_cis = precompute_freqs_cis(
            config.d_model // config.n_heads, config.max_seq_len, config.rope_theta
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_embeddings(idx)
        freqs_cis = self.freqs_cis[:T]

        total_z_loss = 0.0
        for layer in self.layers:
            x, z_loss = layer(x, freqs_cis)
            total_z_loss += z_loss

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # Sumamos z_loss con peso mínimo (1e-4) para estabilidad
            loss = ce_loss + 1e-4 * (total_z_loss / len(self.layers))

        return logits, loss

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        # Parámetros activos por token: Embeddings + Atención + 1 Shared Expert + 2 Routed Experts
        per_layer_expert = sum(p.numel() for p in self.layers[0].moe.shared_expert.parameters())
        inactive_per_layer = (self.config.n_routed_experts - self.config.top_k) * per_layer_expert
        active = total - (inactive_per_layer * self.config.n_layers)
        return total, active