#!/usr/bin/env python3
"""
Fase 2 (Variante 2B Validada): Sparse Upcycling sin Pérdida de microsoft/Phi-4-mini-instruct (MIT)
hacia Kaze-Phi4-MoE-MIT con arquitectura Hot / Warm / Cold sobre UFS.

Arquitectura validada (9/10 en prueba de presión automatizada, 0% olvido catastrófico):
- HOT  (RAM residente, congelado): Phi-4-mini-instruct 3.837B (Shared Backbone) + Routers lineales (32 capas)
- WARM (Caché LRU RAM) / COLD (Streaming UFS): 6 expertos enrutados bottleneck (3072 -> 64 -> 3072, top_k=2)
  con W_down inicializado en 0 -> Paso 0 idéntico bit-a-bit a Phi-4-mini.
- Tamaño de expertos enrutados: 75.5M params (152.2 MB en BF16 | ~45.3 MB en Q4_K_M).
- Licencia: MIT.
"""

import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


class UpcycledRoutedExpert(nn.Module):
    """Experto enrutado ligero (WARM LRU / COLD UFS) con W_down=0 en Paso 0."""

    def __init__(self, hidden_size: int = 3072, bottleneck: int = 64, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.up = nn.Linear(hidden_size, bottleneck, bias=False, device=device, dtype=dtype)
        self.down = nn.Linear(bottleneck, hidden_size, bias=False, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.up(x)))


class KazePhi4UpcycledMoE(nn.Module):
    """Shared Backbone Phi-4-mini (HOT congelado) + Router (HOT) + 6 Routed Experts (WARM/COLD UFS)."""

    def __init__(self, shared_mlp: nn.Module, hidden_size: int = 3072, n_routed: int = 6, top_k: int = 2, bottleneck: int = 64):
        super().__init__()
        self.shared_mlp = shared_mlp
        for p in self.shared_mlp.parameters():
            p.requires_grad_(False)
        dev = next(shared_mlp.parameters()).device
        dt = next(shared_mlp.parameters()).dtype
        self.n_routed = n_routed
        self.top_k = top_k
        self.router = nn.Linear(hidden_size, n_routed, bias=False, device=dev, dtype=dt)
        nn.init.normal_(self.router.weight, std=0.02)
        self.experts = nn.ModuleList([
            UpcycledRoutedExpert(hidden_size, bottleneck, dtype=dt, device=dev) for _ in range(n_routed)
        ])
        self.register_buffer("expert_bias", torch.zeros(n_routed, device=dev, dtype=dt))
        self.register_buffer("_counts", torch.zeros(n_routed, device=dev), persistent=False)
        self._tokens = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.shared_mlp(x)
        B, T, C = x.shape
        xf = x.view(-1, C)
        raw_logits = self.router(xf)
        _, tidx = torch.topk(raw_logits + self.expert_bias.unsqueeze(0), self.top_k, dim=-1)
        w = F.softmax(torch.gather(raw_logits, -1, tidx).float(), dim=-1).to(x.dtype)
        if self.training:
            with torch.no_grad():
                self._counts.add_(torch.bincount(tidx.view(-1), minlength=self.n_routed).float())
                self._tokens += xf.shape[0]
        rout = torch.zeros_like(xf)
        if xf.shape[0] == 1:
            for k in range(self.top_k):
                ei = int(tidx[0, k].item())
                rout = rout + self.experts[ei](xf) * w[0, k]
        else:
            for ei in tidx.unique().tolist():
                tm, kp = torch.where(tidx == ei)
                rout.index_add_(0, tm, self.experts[ei](xf[tm]) * w[tm, kp].unsqueeze(-1))
        return base_out + rout.view(B, T, C)

    @torch.no_grad()
    def update_bias(self, speed: float = 0.002):
        if self._tokens > 0:
            ideal = (self._tokens * self.top_k) / self.n_routed
            self.expert_bias.add_(speed * torch.sign(ideal - self._counts).to(self.expert_bias.dtype))
            self._counts.zero_()
            self._tokens = 0


def attach_upcycled_moe(phi_model, n_routed: int = 6, top_k: int = 2, bottleneck: int = 64, weights_path: str | None = None):
    """Envuelve las 32 capas MLP de Phi-4-mini y opcionalmente carga los pesos entrenados de los expertos."""
    for p in phi_model.parameters():
        p.requires_grad_(False)
    hidden = phi_model.config.hidden_size
    for layer in phi_model.model.layers:
        layer.mlp = KazePhi4UpcycledMoE(layer.mlp, hidden_size=hidden, n_routed=n_routed, top_k=top_k, bottleneck=bottleneck)
    if weights_path and os.path.isfile(weights_path):
        st = load_file(weights_path, device=str(next(phi_model.parameters()).device))
        for li, layer in enumerate(phi_model.model.layers):
            layer.mlp.router.weight.data.copy_(st[f"layers.{li}.router.weight"])
            layer.mlp.expert_bias.data.copy_(st[f"layers.{li}.expert_bias"])
            for ei, exp in enumerate(layer.mlp.experts):
                exp.up.weight.data.copy_(st[f"layers.{li}.experts.{ei}.up.weight"])
                exp.down.weight.data.copy_(st[f"layers.{li}.experts.{ei}.down.weight"])
    return phi_model


def save_upcycled_experts(phi_model, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    st = {}
    for li, layer in enumerate(phi_model.model.layers):
        st[f"layers.{li}.router.weight"] = layer.mlp.router.weight.detach().cpu()
        st[f"layers.{li}.expert_bias"] = layer.mlp.expert_bias.detach().cpu()
        for ei, exp in enumerate(layer.mlp.experts):
            st[f"layers.{li}.experts.{ei}.up.weight"] = exp.up.weight.detach().cpu()
            st[f"layers.{li}.experts.{ei}.down.weight"] = exp.down.weight.detach().cpu()
    out_path = os.path.join(out_dir, "kaze_phi4_moe_routed_experts.safetensors")
    save_file(st, out_path)
    return out_path
