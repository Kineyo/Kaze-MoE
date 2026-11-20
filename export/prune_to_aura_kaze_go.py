#!/usr/bin/env python3
"""
Task-Specific Expert Pruner: Kaze-4.7B-A530M (31 routed experts) -> Aura-Kaze-Go (5-7 routed experts).

Mide qué expertos enrutados se activan más en las tareas objetivo de Aura (shards RAG de Kofu_Mobile,
español escolar, matemáticas y programación) y poda el resto conservando siempre el Shared Expert
y reindexando `gate.weight`, `expert_bias` y `experts.{0..K-1}` sin perder el conocimiento clave.
Licencia: MIT.
"""

import argparse
import json
import os
import shutil
import time
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.kaze_moe import KazeConfig, KazeMoEModel, precompute_rope_cache


DEFAULT_CALIBRATION_PROMPTS = [
    "Hola, ¿quién eres y para qué sirves?",
    "Explica paso a paso qué es la fotosíntesis y por qué produce oxígeno.",
    "Si tengo 4 cajas con 12 libros cada una y regalo 6, ¿cuántos me quedan?",
    "Escribe una función en Python que verifique si una palabra es palíndromo.",
    "¿Qué diferencia hay entre memoria stack y memoria heap en C++?",
    "¿Cuál fue la importancia histórica de la Revolución Industrial?",
    "Dame 3 consejos prácticos para administrar un presupuesto personal.",
]


def load_calibration_texts(rag_shards_path: str | None, max_shards: int = 80) -> list[str]:
    texts = list(DEFAULT_CALIBRATION_PROMPTS)
    if rag_shards_path and os.path.isfile(rag_shards_path):
        with open(rag_shards_path, encoding="utf-8") as f:
            data = json.load(f)
        shards = data.get("shards", {})
        for domain, items in shards.items():
            for item in items:
                title = item.get("title", "")
                content = item.get("content", "")
                if content:
                    texts.append(f"{title}: {content}")
                if len(texts) >= max_shards:
                    return texts
    return texts


@torch.no_grad()
def score_experts(
    model: KazeMoEModel,
    tokenizer: Tokenizer,
    texts: list[str],
    device: str = "cpu",
    max_seq_len: int = 256,
) -> torch.Tensor:
    """Devuelve matriz [n_layers, n_routed_experts] con importancia acumulada por experto."""
    n_layers = model.config.n_layers
    n_routed = model.config.n_routed_experts
    scores = torch.zeros((n_layers, n_routed), dtype=torch.float64)

    hooks = []
    for layer_idx, layer in enumerate(model.layers):
        def make_hook(li: int, moe_mod):
            def _hook(_mod, inp, _out):
                x = inp[0].view(-1, inp[0].shape[-1])
                raw_logits = moe_mod.gate(x)
                routing = raw_logits + moe_mod.expert_bias.unsqueeze(0)
                top_vals, top_idx = torch.topk(routing, moe_mod.top_k, dim=-1)
                weights = torch.softmax(torch.gather(raw_logits, -1, top_idx).float(), dim=-1)
                for k in range(moe_mod.top_k):
                    idx_k = top_idx[:, k].cpu()
                    w_k = weights[:, k].double().cpu()
                    scores[li].index_add_(0, idx_k, w_k)
            return _hook

        hooks.append(layer.moe.register_forward_hook(make_hook(layer_idx, layer.moe)))

    model.eval()
    for txt in texts:
        ids = tokenizer.encode(txt).ids[:max_seq_len]
        if len(ids) < 2:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        model(x)

    for h in hooks:
        h.remove()
    return scores


def prune_state_dict(
    state: dict[str, torch.Tensor],
    scores: torch.Tensor,
    keep_experts: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    n_layers, n_routed = scores.shape
    assert 1 <= keep_experts <= n_routed, f"keep_experts={keep_experts} fuera de rango [1, {n_routed}]"

    pruned: dict[str, torch.Tensor] = {}
    selected_map: dict[str, list[int]] = {}

    for li in range(n_layers):
        top_ids = torch.topk(scores[li], keep_experts).indices.tolist()
        top_ids.sort()
        selected_map[f"layer_{li}"] = top_ids

    for k, v in state.items():
        if k == "lm_head.weight":
            continue

        # Detectar capas de expertos enrutados: layers.{li}.moe.experts.{ei}.*
        if ".moe.experts." in k:
            parts = k.split(".")
            li = int(parts[1])
            ei = int(parts[4])
            kept = selected_map[f"layer_{li}"]
            if ei in kept:
                new_ei = kept.index(ei)
                parts[4] = str(new_ei)
                pruned[".".join(parts)] = v.clone()
            continue

        # Rebanar gate.weight [n_routed, d_model] y expert_bias [n_routed]
        if k.endswith(".moe.gate.weight") or k.endswith(".moe.expert_bias"):
            li = int(k.split(".")[1])
            idx_t = torch.tensor(selected_map[f"layer_{li}"], dtype=torch.long, device=v.device)
            pruned[k] = v.index_select(0, idx_t).clone()
            continue

        pruned[k] = v.clone()

    report = {
        "original_routed_experts": int(n_routed),
        "pruned_routed_experts": int(keep_experts),
        "selected_experts_per_layer": selected_map,
    }
    return pruned, report


def main():
    parser = argparse.ArgumentParser(description="Poda Kaze-4.7B-A530M hacia Aura-Kaze-Go (5-7 expertos)")
    parser.add_argument("--checkpoint", required=True, help="Ruta a kaze_4.7b_*.safetensors")
    parser.add_argument("--config", required=True, help="Ruta a config.json de Kaze")
    parser.add_argument("--tokenizer", required=True, help="Ruta a tokenizer.json")
    parser.add_argument("--output-dir", required=True, help="Carpeta de salida para Aura-Kaze-Go")
    parser.add_argument("--keep-experts", type=int, default=6, help="Expertos enrutados a conservar (5-7 recomendado)")
    parser.add_argument("--rag-shards", default="", help="Ruta opcional a kofu_knowledge_shards.json para calibrar")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.config, encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = KazeConfig(**cfg_dict)
    tok = Tokenizer.from_file(args.tokenizer)

    with torch.device("meta"):
        model = KazeMoEModel(cfg)
    raw_state = load_file(args.checkpoint, device=args.device)
    if "lm_head.weight" not in raw_state and "tok_embeddings.weight" in raw_state:
        raw_state["lm_head.weight"] = raw_state["tok_embeddings.weight"]
    model.load_state_dict(raw_state, strict=False, assign=True)
    model.lm_head.weight = model.tok_embeddings.weight
    # Materializar buffers no persistentes (RoPE y contadores MoE) que quedan en meta device
    freqs_cos, freqs_sin = precompute_rope_cache(
        cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta
    )
    model.freqs_cos = freqs_cos.to(args.device)
    model.freqs_sin = freqs_sin.to(args.device)
    for layer in model.layers:
        layer.moe._step_counts = torch.zeros(cfg.n_routed_experts, device=args.device)
        layer.moe._step_tokens = 0
    model = model.to(device=args.device)

    texts = load_calibration_texts(args.rag_shards)
    print(f"Calibrando importancia de expertos sobre {len(texts)} textos en {args.device}...")
    scores = score_experts(model, tok, texts, device=args.device)

    pruned_state, report = prune_state_dict(raw_state, scores, args.keep_experts)
    out_st = os.path.join(args.output_dir, f"aura_kaze_go_{args.keep_experts}exp_raw.safetensors")
    save_file({k: v.cpu() for k, v in pruned_state.items()}, out_st)

    new_cfg = dict(cfg_dict)
    new_cfg["n_routed_experts"] = args.keep_experts
    new_cfg["top_k"] = min(cfg_dict.get("top_k", 2), args.keep_experts)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(new_cfg, f, indent=2)
    with open(os.path.join(args.output_dir, "pruning_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    shutil.copy2(args.tokenizer, os.path.join(args.output_dir, "tokenizer.json"))

    orig_params = sum(v.numel() for k, v in raw_state.items() if k != "lm_head.weight")
    new_params = sum(v.numel() for v in pruned_state.values())
    print(f"Poda completada en {time.time()-t0:.1f}s:")
    print(f"  Parámetros: {orig_params/1e9:.3f}B -> {new_params/1e9:.3f}B ({100*(1-new_params/orig_params):.1f}% reducción)")
    print(f"  Guardado en: {out_st}")


if __name__ == "__main__":
    main()
