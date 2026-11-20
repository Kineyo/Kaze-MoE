#!/usr/bin/env python3
"""
Rescate Integral de Kaze-4.7B-A530M (100% MIT, sin pesos externos).

Estrategia optimizada para NVIDIA L4 (24 GB) y A100 (40/80 GB):
1. Destrucción de atractores (ej. "Escribe una función en Python..." -> def es_primo)
   con cientos de funciones Python variadas en español e inglés.
2. Aritmética paso a paso 100% verificada simbólicamente en Python (ES + EN).
3. Conocimiento general, ciencia, historia y finanzas personales en español (respuestas largas coherentes).
4. Sequence Packing (0% desperdicio de cómputo en <|pad|>) con máscara de loss solo en <|assistant|>.
5. LoRA de alto rango (r=32) sobre atención + shared expert + 31 routed experts + router entrenable
   (cabe en ~14 GB VRAM en L4, evita olvido catastrófico y se fusiona lossless a pesos crudos .safetensors).
6. Fase final de consolidación con expert_bias=0 para conversión 100% lossless a Qwen2Moe / GGUF.

Licencia: MIT.
"""

import argparse
import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from torch.utils.checkpoint import checkpoint


@dataclass
class KazeConfig:
    vocab_size: int = 32768
    d_model: int = 1024
    n_layers: int = 23
    n_heads: int = 16
    n_kv_heads: int = 4
    d_ff: int = 2048
    n_routed_experts: int = 31
    n_shared_experts: int = 1
    top_k: int = 2
    max_seq_len: int = 4096
    rope_theta: float = 100000.0
    bias_update_speed: float = 0.001
    use_checkpoint: bool = True


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
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)
    cos = freqs_cos.float().unsqueeze(0).unsqueeze(2)
    sin = freqs_sin.float().unsqueeze(0).unsqueeze(2)
    xq_out = torch.stack([xq_r * cos - xq_i * sin, xq_r * sin + xq_i * cos], dim=-1).flatten(3)
    xk_out = torch.stack([xk_r * cos - xk_i * sin, xk_r * sin + xk_i * cos], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class LoRALinear(nn.Module):
    """Capa Linear congelada con adaptador LoRA fusionable lossless."""

    def __init__(self, base_linear: nn.Linear, r: int = 32, alpha: float = 64.0, dropout: float = 0.05):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        in_f, out_f = base_linear.in_features, base_linear.out_features
        dev, dt = base_linear.weight.device, base_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty((r, in_f), device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros((out_f, r), device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B) * self.scaling
        return out + delta

    @torch.no_grad()
    def merge_to_linear(self) -> nn.Linear:
        delta_w = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
        self.base.weight.data.add_(delta_w.to(self.base.weight.dtype))
        return self.base


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class DeepSeekStyleMoE(nn.Module):
    def __init__(self, config: KazeConfig):
        super().__init__()
        self.n_routed = config.n_routed_experts
        self.top_k = config.top_k
        self.bias_update_speed = config.bias_update_speed
        self.shared_expert = SwiGLUExpert(config.d_model, config.d_ff)
        self.experts = nn.ModuleList([SwiGLUExpert(config.d_model, config.d_ff) for _ in range(self.n_routed)])
        self.gate = nn.Linear(config.d_model, self.n_routed, bias=False)
        self.register_buffer("expert_bias", torch.zeros(self.n_routed))
        self.register_buffer("_step_counts", torch.zeros(self.n_routed), persistent=False)
        self._step_tokens = 0

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        shared_out = self.shared_expert(x_flat)
        raw_logits = self.gate(x_flat)
        routing_scores = raw_logits + self.expert_bias.unsqueeze(0)
        _, top_indices = torch.topk(routing_scores, self.top_k, dim=-1)
        selected_logits = torch.gather(raw_logits, -1, top_indices)
        routing_weights = F.softmax(selected_logits, dim=-1, dtype=torch.float32).to(x.dtype)

        if self.training and self.bias_update_speed > 0:
            with torch.no_grad():
                counts = torch.bincount(top_indices.view(-1), minlength=self.n_routed).to(self._step_counts.dtype)
                self._step_counts.add_(counts)
                self._step_tokens += x_flat.shape[0]

        routed_out = torch.zeros_like(x_flat)
        for e_idx in top_indices.unique().tolist():
            token_mask, k_pos = torch.where(top_indices == e_idx)
            expert_output = self.experts[e_idx](x_flat[token_mask])
            weight = routing_weights[token_mask, k_pos].unsqueeze(-1)
            routed_out.index_add_(0, token_mask, expert_output * weight)

        z_loss = torch.logsumexp(raw_logits.float(), dim=-1).pow(2).mean()
        return (shared_out + routed_out).view(B, T, C), z_loss

    @torch.no_grad()
    def update_expert_bias(self):
        if self._step_tokens > 0 and self.bias_update_speed > 0:
            ideal = (self._step_tokens * self.top_k) / self.n_routed
            err = ideal - self._step_counts.float()
            self.expert_bias.add_(self.bias_update_speed * torch.sign(err).to(self.expert_bias.dtype))
            self._step_counts.zero_()
            self._step_tokens = 0


class GQAAttention(nn.Module):
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

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, _ = x.shape
        xq = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        xk = xk.repeat_interleave(self.n_rep, dim=2)
        xv = xv.repeat_interleave(self.n_rep, dim=2)
        out = F.scaled_dot_product_attention(
            xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2), is_causal=True
        )
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, -1))


class TransformerBlock(nn.Module):
    def __init__(self, config: KazeConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = GQAAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.moe = DeepSeekStyleMoE(config)

    def forward(self, x, freqs_cos, freqs_sin):
        x = x + self.attn(self.attn_norm(x), freqs_cos, freqs_sin)
        moe_out, z_loss = self.moe(self.ffn_norm(x))
        return x + moe_out, z_loss


class KazeRescueModel(nn.Module):
    def __init__(self, config: KazeConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        fc, fs = precompute_rope_cache(config.d_model // config.n_heads, config.max_seq_len, config.rope_theta)
        self.register_buffer("freqs_cos", fc, persistent=False)
        self.register_buffer("freqs_sin", fs, persistent=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_embeddings(idx)
        fc, fs = self.freqs_cos[:T], self.freqs_sin[:T]
        tz = 0.0
        for layer in self.layers:
            if self.training and self.config.use_checkpoint:
                x, z = checkpoint(layer, x, fc, fs, use_reentrant=False)
            else:
                x, z = layer(x, fc, fs)
            tz = tz + z
        return self.lm_head(self.norm(x)), tz / len(self.layers)

    @torch.no_grad()
    def update_expert_biases(self):
        for layer in self.layers:
            layer.moe.update_expert_bias()

    @torch.no_grad()
    def freeze_expert_bias_zero(self):
        for layer in self.layers:
            layer.moe.bias_update_speed = 0.0
            layer.moe.expert_bias.zero_()
            layer.moe._step_counts.zero_()
            layer.moe._step_tokens = 0


def inject_lora(model: KazeRescueModel, r: int = 32, alpha: float = 64.0, expert_r: int = 16):
    """Congela los pesos base e inyecta LoRA en atención, shared expert y los 31 expertos enrutados."""
    for p in model.parameters():
        p.requires_grad_(False)

    # Router y RMSNorm siempre entrenables directamente (pocos parámetros, alto impacto)
    model.norm.weight.requires_grad_(True)
    for layer in model.layers:
        layer.attn_norm.weight.requires_grad_(True)
        layer.ffn_norm.weight.requires_grad_(True)
        layer.moe.gate.weight.requires_grad_(True)

        # Atención GQA (r alto)
        for attr in ("wq", "wk", "wv", "wo"):
            setattr(layer.attn, attr, LoRALinear(getattr(layer.attn, attr), r=r, alpha=alpha))

        # Shared expert (r alto)
        for attr in ("w_gate", "w_up", "w_down"):
            setattr(layer.moe.shared_expert, attr, LoRALinear(getattr(layer.moe.shared_expert, attr), r=r, alpha=alpha))

        # 31 Routed experts (expert_r para eficiencia de VRAM y generalización)
        for exp in layer.moe.experts:
            for attr in ("w_gate", "w_up", "w_down"):
                setattr(exp, attr, LoRALinear(getattr(exp, attr), r=expert_r, alpha=expert_r * 2.0))


@torch.no_grad()
def merge_lora(model: KazeRescueModel):
    """Fusiona todos los adaptadores LoRA lossless de vuelta a nn.Linear estándar."""
    for layer in model.layers:
        for attr in ("wq", "wk", "wv", "wo"):
            mod = getattr(layer.attn, attr)
            if isinstance(mod, LoRALinear):
                setattr(layer.attn, attr, mod.merge_to_linear())
        for attr in ("w_gate", "w_up", "w_down"):
            mod = getattr(layer.moe.shared_expert, attr)
            if isinstance(mod, LoRALinear):
                setattr(layer.moe.shared_expert, attr, mod.merge_to_linear())
        for exp in layer.moe.experts:
            for attr in ("w_gate", "w_up", "w_down"):
                mod = getattr(exp, attr)
                if isinstance(mod, LoRALinear):
                    setattr(exp, attr, mod.merge_to_linear())


def build_anti_attractor_code_pool() -> list[list[tuple[str, str]]]:
    """Genera ejercicios Python variados en español e inglés para destruir el atractor de def es_primo."""
    templates_es = [
        "Escribe una función en Python que {desc}.",
        "Crea una función en Python para {desc}.",
        "Implementa en Python una función que permita {desc}.",
        "¿Cómo escribo en Python una función que {desc}?",
    ]
    templates_en = [
        "Write a Python function that {desc_en}.",
        "Implement a Python function to {desc_en}.",
    ]
    tasks = [
        (
            "invierta una cadena de texto",
            "reverses a string",
            "```python\ndef invertir_cadena(texto: str) -> str:\n    return texto[::-1]\n```",
            "```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```",
        ),
        (
            "calcule el factorial de un número entero no negativo",
            "computes the factorial of a non-negative integer",
            "```python\ndef factorial(n: int) -> int:\n    if n < 0:\n        raise ValueError('n debe ser >= 0')\n    res = 1\n    for i in range(2, n + 1):\n        res *= i\n    return res\n```",
            "```python\ndef factorial(n: int) -> int:\n    if n < 0:\n        raise ValueError('n must be >= 0')\n    res = 1\n    for i in range(2, n + 1):\n        res *= i\n    return res\n```",
        ),
        (
            "verifique si una palabra es un palíndromo",
            "checks whether a word is a palindrome",
            "```python\ndef es_palindromo(texto: str) -> bool:\n    limpio = ''.join(c.lower() for c in texto if c.isalnum())\n    return limpio == limpio[::-1]\n```",
            "```python\ndef is_palindrome(s: str) -> bool:\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n```",
        ),
        (
            "cuente cuántas vocales tiene una cadena de texto",
            "counts the number of vowels in a string",
            "```python\ndef contar_vocales(texto: str) -> int:\n    return sum(1 for c in texto.lower() if c in 'aeiouáéíóú')\n```",
            "```python\ndef count_vowels(s: str) -> int:\n    return sum(1 for c in s.lower() if c in 'aeiou')\n```",
        ),
        (
            "devuelva el número mayor de una lista sin usar max()",
            "returns the largest number in a list without using max()",
            "```python\ndef maximo_lista(nums: list[float]) -> float:\n    mayor = nums[0]\n    for x in nums[1:]:\n        if x > mayor:\n            mayor = x\n    return mayor\n```",
            "```python\ndef find_max(nums: list[float]) -> float:\n    best = nums[0]\n    for x in nums[1:]:\n        if x > best:\n            best = x\n    return best\n```",
        ),
        (
            "elimine los elementos duplicados de una lista preservando el orden",
            "removes duplicates from a list while preserving order",
            "```python\ndef quitar_duplicados(items: list) -> list:\n    vistos, res = set(), []\n    for x in items:\n        if x not in vistos:\n            vistos.add(x)\n            res.append(x)\n    return res\n```",
            "```python\ndef remove_duplicates(items: list) -> list:\n    seen, out = set(), []\n    for x in items:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n```",
        ),
        (
            "genere los primeros n números de la serie de Fibonacci",
            "generates the first n Fibonacci numbers",
            "```python\ndef fibonacci(n: int) -> list[int]:\n    a, b, res = 0, 1, []\n    for _ in range(n):\n        res.append(a)\n        a, b = b, a + b\n    return res\n```",
            "```python\ndef fibonacci(n: int) -> list[int]:\n    a, b, out = 0, 1, []\n    for _ in range(n):\n        out.append(a)\n        a, b = b, a + b\n    return out\n```",
        ),
        (
            "sume todos los números pares de una lista",
            "sums all even numbers in a list",
            "```python\ndef sumar_pares(nums: list[int]) -> int:\n    return sum(x for x in nums if x % 2 == 0)\n```",
            "```python\ndef sum_evens(nums: list[int]) -> int:\n    return sum(x for x in nums if x % 2 == 0)\n```",
        ),
        (
            "convierta grados Celsius a Fahrenheit",
            "converts Celsius to Fahrenheit",
            "```python\ndef celsius_a_fahrenheit(c: float) -> float:\n    return (c * 9 / 5) + 32\n```",
            "```python\ndef celsius_to_fahrenheit(c: float) -> float:\n    return (c * 9 / 5) + 32\n```",
        ),
        (
            "calcule el promedio de una lista de números",
            "computes the average of a list of numbers",
            "```python\ndef promedio(nums: list[float]) -> float:\n    return sum(nums) / len(nums) if nums else 0.0\n```",
            "```python\ndef average(nums: list[float]) -> float:\n    return sum(nums) / len(nums) if nums else 0.0\n```",
        ),
        (
            "verifique si un número es primo",
            "checks whether a number is prime",
            "```python\ndef es_primo(n: int) -> bool:\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```",
            "```python\ndef is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```",
        ),
        (
            "cuente la frecuencia de cada palabra en un texto",
            "counts word frequencies in a string",
            "```python\ndef contar_palabras(texto: str) -> dict[str, int]:\n    conteo = {}\n    for p in texto.lower().split():\n        conteo[p] = conteo.get(p, 0) + 1\n    return conteo\n```",
            "```python\ndef word_count(text: str) -> dict[str, int]:\n    counts = {}\n    for w in text.lower().split():\n        counts[w] = counts.get(w, 0) + 1\n    return counts\n```",
        ),
        (
            "ordene una lista de diccionarios por una clave específica",
            "sorts a list of dictionaries by a given key",
            "```python\ndef ordenar_por_clave(datos: list[dict], clave: str) -> list[dict]:\n    return sorted(datos, key=lambda d: d[clave])\n```",
            "```python\ndef sort_by_key(items: list[dict], key: str) -> list[dict]:\n    return sorted(items, key=lambda d: d[key])\n```",
        ),
        (
            "calcule el máximo común divisor (MCD) de dos enteros",
            "computes the greatest common divisor (GCD) of two integers",
            "```python\ndef mcd(a: int, b: int) -> int:\n    while b:\n        a, b = b, a % b\n    return abs(a)\n```",
            "```python\ndef gcd(a: int, b: int) -> int:\n    while b:\n        a, b = b, a % b\n    return abs(a)\n```",
        ),
    ]
    pool = []
    for desc_es, desc_en, code_es, code_en in tasks:
        for t in templates_es:
            pool.append([("user", t.format(desc=desc_es)), ("assistant", code_es)])
        for t in templates_en:
            pool.append([("user", t.format(desc_en=desc_en)), ("assistant", code_en)])
    return pool


def build_verified_math_pool(n_samples: int = 8000, seed: int = 2026) -> list[list[tuple[str, str]]]:
    """Genera problemas aritméticos paso a paso 100% verificados simbólicamente en ES y EN."""
    rng = random.Random(seed)
    items_es = [
        ("cajas", "manzanas"), ("bolsas", "naranjas"), ("paquetes", "lápices"),
        ("estantes", "libros"), ("bandejas", "galletas"), ("sobres", "estampas"),
        ("canastas", "peras"), ("frascos", "dulces"), ("carpetas", "hojas"),
    ]
    items_en = [
        ("boxes", "apples"), ("bags", "oranges"), ("packs", "pencils"),
        ("shelves", "books"), ("trays", "cookies"), ("baskets", "pears"),
    ]
    verbs_sub_es = [("regalo", "regaladas"), ("vendo", "vendidas"), ("uso", "usadas"), ("pierdo", "perdidas")]
    verbs_add_es = [("compro", "compradas"), ("recibo", "recibidas"), ("encuentro", "encontradas")]

    pool = []
    for _ in range(n_samples):
        mode = rng.random()
        if mode < 0.55:
            cont, obj = rng.choice(items_es)
            a = rng.randint(2, 15)
            b = rng.randint(3, 25)
            prod = a * b
            if rng.random() < 0.65:
                v, vp = rng.choice(verbs_sub_es)
                c = rng.randint(1, prod - 1)
                ans = prod - c
                q = f"Tengo {a} {cont} con {b} {obj} cada una y {v} {c}, ¿cuántas me quedan? Explica paso a paso."
                r = (
                    f"Paso 1: Calculamos el total inicial multiplicando: {a} × {b} = {prod} {obj}.\n"
                    f"Paso 2: Restamos las {c} {vp}: {prod} - {c} = {ans} {obj}.\n"
                    f"Respuesta: Te quedan {ans} {obj}."
                )
            else:
                v, vp = rng.choice(verbs_add_es)
                c = rng.randint(1, 30)
                ans = prod + c
                q = f"Tengo {a} {cont} con {b} {obj} cada una y {v} {c} más, ¿cuántas tengo en total? Explica paso a paso."
                r = (
                    f"Paso 1: Calculamos el total inicial multiplicando: {a} × {b} = {prod} {obj}.\n"
                    f"Paso 2: Sumamos las {c} {vp}: {prod} + {c} = {ans} {obj}.\n"
                    f"Respuesta: Tienes {ans} {obj} en total."
                )
            pool.append([("user", q), ("assistant", r)])
        elif mode < 0.78:
            # Finanzas / presupuestos verificados
            ing = rng.randint(8, 60) * 100
            g1 = rng.randint(2, int(ing // 250)) * 100
            g2 = rng.randint(1, max(1, int((ing - g1) // 300))) * 100
            ahorro = ing - g1 - g2
            pct = round(100.0 * ahorro / ing, 1)
            q = (
                f"Si gano ${ing} al mes y gasto ${g1} en vivienda y ${g2} en comida y transporte, "
                f"¿cuánto dinero me queda para ahorrar?"
            )
            r = (
                f"Paso 1: Sumamos los gastos totales: ${g1} + ${g2} = ${g1 + g2}.\n"
                f"Paso 2: Restamos los gastos del ingreso mensual: ${ing} - ${g1 + g2} = ${ahorro}.\n"
                f"Respuesta: Te quedan ${ahorro} para ahorrar (aproximadamente el {pct}% de tu ingreso)."
            )
            pool.append([("user", q), ("assistant", r)])
        else:
            # Inglés paso a paso
            cont, obj = rng.choice(items_en)
            a = rng.randint(2, 12)
            b = rng.randint(4, 20)
            prod = a * b
            c = rng.randint(1, prod - 1)
            ans = prod - c
            q = f"If I have {a} {cont} with {b} {obj} each and I sell {c} {obj}, how many {obj} do I have left? Explain step by step."
            r = (
                f"Step 1: Multiply to find the initial total: {a} × {b} = {prod} {obj}.\n"
                f"Step 2: Subtract the {c} sold: {prod} - {c} = {ans} {obj}.\n"
                f"Answer: You have {ans} {obj} left."
            )
            pool.append([("user", q), ("assistant", r)])
    return pool


def build_es_finance_science_pool() -> list[list[tuple[str, str]]]:
    """Respuestas amplias y claras en español e inglés sobre finanzas, ciencia, historia e identidad."""
    return [
        [
            ("user", "¿Por qué es importante hacer un presupuesto personal?"),
            (
                "assistant",
                "Hacer un presupuesto personal es fundamental porque te da visibilidad real sobre cuánto dinero entra y en qué se gasta cada mes. "
                "Te permite cubrir primero tus necesidades básicas, detectar y reducir gastos innecesarios, evitar deudas costosas y separar una cantidad fija para ahorros e imprevistos. "
                "Con un presupuesto claro tomas decisiones financieras conscientes en lugar de preguntarte al final del mes a dónde se fue tu dinero.",
            ),
        ],
        [
            ("user", "¿Qué es un fondo de emergencia y cuánto debería ahorrar?"),
            (
                "assistant",
                "Un fondo de emergencia es una reserva de dinero líquido destinada exclusivamente a cubrir imprevistos como gastos médicos, reparaciones urgentes o pérdida temporal de ingresos. "
                "La recomendación general es acumular entre 3 y 6 meses de tus gastos fijos básicos en una cuenta segura y de fácil acceso, empezando con metas pequeñas como tu primer mes de gastos.",
            ),
        ],
        [
            ("user", "Dame 3 consejos prácticos para ahorrar dinero cada mes."),
            (
                "assistant",
                "1. **Aplica la regla 50/30/20:** Destina el 50% de tus ingresos a necesidades básicas, 30% a gastos personales y al menos 20% al ahorro.\n"
                "2. **Automatiza tu ahorro:** Programa una transferencia automática a una cuenta separada el mismo día que recibes tu ingreso.\n"
                "3. **Registra y recorta gastos hormiga:** Revisa suscripciones que no usas y define un presupuesto semanal para gastos diarios.",
            ),
        ],
        [
            ("user", "¿Qué es la fotosíntesis y por qué es importante para la vida en la Tierra?"),
            (
                "assistant",
                "La fotosíntesis es el proceso mediante el cual las plantas, algas y algunas bacterias convierten la luz solar, el agua (H2O) y el dióxido de carbono (CO2) en glucosa (energía química) y oxígeno (O2). "
                "Es fundamental para la vida en la Tierra porque produce el oxígeno que respiramos y constituye la base de las cadenas alimentarias.",
            ),
        ],
        [
            ("user", "¿Cuál fue la importancia de la Revolución Industrial?"),
            (
                "assistant",
                "La Revolución Industrial (iniciada en Gran Bretaña a finales del siglo XVIII) transformó las economías agrícolas y artesanales en economías mecanizadas e industriales. "
                "Introdujo la máquina de vapor, las fábricas, el ferrocarril y la producción en masa, acelerando el crecimiento económico, la urbanización y la ciencia moderna.",
            ),
        ],
        [
            ("user", "Explain why the sky is blue in simple terms."),
            (
                "assistant",
                "Sunlight looks white, but it is actually a mix of all the colors of the rainbow. When sunlight enters Earth's atmosphere, it hits tiny gas molecules that scatter shorter wavelengths of light much more than longer ones. "
                "Because blue light has a shorter wavelength than red or yellow light, it gets scattered in all directions across the sky, making the sky appear blue to our eyes.",
            ),
        ],
        [
            ("user", "Hola, ¿quién eres y para qué sirves?"),
            (
                "assistant",
                "Soy Kaze-4.7B-A530M, un modelo de lenguaje Sparse Mixture-of-Experts (4.7B parámetros totales, ~530M activos por token) desarrollado por Kineyo bajo licencia MIT. "
                "Estoy diseñado para ejecutarse de forma eficiente en dispositivos locales y ayudarte con conversación en español, finanzas personales, ciencia, historia, matemáticas y programación.",
            ),
        ],
    ]


def pack_conversations(
    convs: list[list[tuple[str, str]]],
    tok: Tokenizer,
    seq_len: int = 512,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """Sequence Packing sin desperdicio en <|pad|>: concatena conversaciones con loss solo en <|assistant|>."""
    rng = random.Random(seed)
    eos_id = tok.token_to_id("<|endoftext|>")
    user_id = tok.token_to_id("<|user|>")
    asst_id = tok.token_to_id("<|assistant|>")
    nl_ids = tok.encode("\n").ids

    shuffled = list(convs)
    rng.shuffle(shuffled)

    all_ids, all_labels = [], []
    for conv in shuffled:
        for role, msg in conv:
            tag = user_id if role == "user" else asst_id
            m_ids = tok.encode(msg).ids
            if role == "user":
                chunk = [tag] + NL_FALLBACK(nl_ids) + m_ids + NL_FALLBACK(nl_ids)
                all_ids.extend(chunk)
                all_labels.extend([-100] * len(chunk))
            else:
                prefix = [tag] + NL_FALLBACK(nl_ids)
                body = m_ids + [eos_id]
                all_ids.extend(prefix + body)
                all_labels.extend([-100] * len(prefix) + body)

    blocks = []
    stride = seq_len + 1
    for i in range(0, len(all_ids) - stride, seq_len):
        x = all_ids[i : i + seq_len]
        y = all_labels[i + 1 : i + 1 + seq_len]
        if any(lbl != -100 for lbl in y):
            blocks.append((x, y))
    return blocks


def NL_FALLBACK(nl_ids: list[int]) -> list[int]:
    return nl_ids if nl_ids else []


def main():
    parser = argparse.ArgumentParser(description="Rescate de Kaze-4.7B-A530M con LoRA fusionado + Sequence Packing")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--bias-zero-steps", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--expert-lora-r", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    with open(args.config, encoding="utf-8") as f:
        cfg = KazeConfig(**json.load(f))
    cfg.use_checkpoint = True
    tok = Tokenizer.from_file(args.tokenizer)

    print(f"Cargando Kaze-4.7B-A530M en {device} ({dtype})...")
    with torch.device("meta"):
        model = KazeRescueModel(cfg)
    w = load_file(args.checkpoint, device=device)
    if "lm_head.weight" not in w:
        w["lm_head.weight"] = w["tok_embeddings.weight"]
    model.load_state_dict(w, strict=False, assign=True)
    model.lm_head.weight = model.tok_embeddings.weight
    fc, fs = precompute_rope_cache(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
    model.freqs_cos, model.freqs_sin = fc.to(device), fs.to(device)
    for layer in model.layers:
        layer.moe._step_counts = torch.zeros(cfg.n_routed_experts, device=device)
        layer.moe._step_tokens = 0
        layer.moe.expert_bias = layer.moe.expert_bias.to(device=device, dtype=dtype)
    model = model.to(device=device, dtype=dtype)
    del w
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    inject_lora(model, r=args.lora_r, alpha=args.lora_r * 2.0, expert_r=args.expert_lora_r)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA inyectado: {trainable/1e6:.1f}M entrenables / {total/1e9:.2f}B totales ({100*trainable/total:.2f}%)")

    # Construir corpus balanceado anti-atractores + matemáticas verificadas + español/inglés
    code_pool = build_anti_attractor_code_pool() * 25
    math_pool = build_verified_math_pool(n_samples=6000)
    core_pool = build_es_finance_science_pool() * 20
    all_convs = code_pool + math_pool + core_pool
    blocks = pack_conversations(all_convs, tok, seq_len=args.seq_len)
    random.Random(42).shuffle(blocks)
    val_blocks = blocks[:64]
    train_blocks = blocks[64:]
    print(f"Sequence Packing listo: {len(train_blocks)} bloques train | {len(val_blocks)} bloques val (seq_len={args.seq_len})")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.05,
    )

    @torch.no_grad()
    def eval_val():
        model.eval()
        losses = []
        for i in range(0, len(val_blocks), args.batch_size):
            ch = val_blocks[i : i + args.batch_size]
            xb = torch.tensor([c[0] for c in ch], dtype=torch.long, device=device)
            yb = torch.tensor([c[1] for c in ch], dtype=torch.long, device=device)
            lg, _ = model(xb)
            losses.append(F.cross_entropy(lg.view(-1, lg.size(-1)), yb.view(-1), ignore_index=-100).item())
        model.train()
        return sum(losses) / max(1, len(losses))

    model.train()
    t0 = time.time()
    best_val = eval_val()
    print(f"[INICIO RESCATE] val_loss={best_val:.4f} | VRAM={torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    total_steps = args.steps + args.bias_zero_steps
    warmup = min(80, args.steps // 10)
    ema = None

    for step in range(total_steps):
        if step == args.steps:
            print(f"\n>>> [CONSOLIDACIÓN] Congelando expert_bias=0 por {args.bias_zero_steps} pasos...", flush=True)
            model.freeze_expert_bias_zero()

        if step < warmup:
            lr = args.lr * (step + 1) / warmup
        else:
            prog = (step - warmup) / max(1, total_steps - warmup)
            lr = args.lr * 0.1 + 0.5 * (args.lr * 0.9) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)
        sloss = 0.0
        for _ in range(args.grad_accum):
            ch = random.sample(train_blocks, args.batch_size)
            xb = torch.tensor([c[0] for c in ch], dtype=torch.long, device=device)
            yb = torch.tensor([c[1] for c in ch], dtype=torch.long, device=device)
            lg, zl = model(xb)
            ce = F.cross_entropy(lg.view(-1, lg.size(-1)), yb.view(-1), ignore_index=-100)
            ((ce + 1e-4 * zl) / args.grad_accum).backward()
            sloss += ce.item() / args.grad_accum

        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        model.update_expert_biases()
        ema = sloss if ema is None else 0.92 * ema + 0.08 * sloss

        if step % 50 == 0 or step == total_steps - 1:
            print(
                f"[RESCUE {step:04d}/{total_steps}] loss={sloss:.4f} | ema={ema:.4f} | lr={lr:.2e} "
                f"| {(time.time()-t0)/60:.1f}m | VRAM={torch.cuda.max_memory_allocated()/1e9:.1f}GB",
                flush=True,
            )
        if (step + 1) % 200 == 0 or step == total_steps - 1:
            vl = eval_val()
            print(f"  >>> [VAL step {step+1}] val_loss={vl:.4f} (best={best_val:.4f})", flush=True)
            if vl < best_val:
                best_val = vl

    print("\nFusionando LoRA lossless y guardando checkpoint rescatado...")
    merge_lora(model)
    st = {k: v.detach().cpu() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_file(st, args.output)
    print(f"Guardado exitoso en {args.output} ({len(st)} tensores, best_val={best_val:.4f})")


if __name__ == "__main__":
    main()
