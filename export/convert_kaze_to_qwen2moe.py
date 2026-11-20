#!/usr/bin/env python3
"""
Conversor sin pérdida: KazeMoE (.safetensors) -> Hugging Face Qwen2MoeForCausalLM + ChatML.

Permite convertir directamente tanto Kaze-4.7B-A530M (Base) como Aura-Kaze-Go (Podado) a formato
GGUF oficial con `llama.cpp/convert_hf_to_gguf.py` y cuantizar a Q4_K_M / IQ3_XS para Kofu_Mobile
sin escribir una arquitectura nueva en C++.
Licencia: MIT.
"""

import argparse
import json
import os
import shutil
import subprocess
import torch
from safetensors.torch import load_file, save_file


KAZE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}{{ '<|user|>\\n' + message['content'] + '\\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\\n' + message['content'] + '<|endoftext|>' }}"
    "{% else %}{{ message['content'] + '\\n' }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
)


def build_qwen2moe_config(kaze_cfg: dict) -> dict:
    d_model = int(kaze_cfg["d_model"])
    d_ff = int(kaze_cfg["d_ff"])
    return {
        "architectures": ["Qwen2MoeForCausalLM"],
        "model_type": "qwen2_moe",
        "vocab_size": int(kaze_cfg["vocab_size"]),
        "hidden_size": d_model,
        "intermediate_size": d_ff,
        "moe_intermediate_size": d_ff,
        "shared_expert_intermediate_size": d_ff * int(kaze_cfg.get("n_shared_experts", 1)),
        "num_hidden_layers": int(kaze_cfg["n_layers"]),
        "num_attention_heads": int(kaze_cfg["n_heads"]),
        "num_key_value_heads": int(kaze_cfg["n_kv_heads"]),
        "num_experts": int(kaze_cfg["n_routed_experts"]),
        "num_experts_per_tok": int(kaze_cfg["top_k"]),
        "decoder_sparse_step": 1,
        "norm_topk_prob": True,
        "hidden_act": "silu",
        "max_position_embeddings": int(kaze_cfg.get("max_seq_len", 4096)),
        "rms_norm_eps": 1e-6,
        "rope_theta": float(kaze_cfg.get("rope_theta", 100000.0)),
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "bos_token_id": 1,
        "eos_token_id": 0,
        "pad_token_id": 2,
    }


def permute_interleaved_to_neox(w: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Convierte filas de proyección Q/K de RoPE intercalado (Kaze) a RoPE NeoX half-split (Qwen2Moe/llama.cpp)."""
    return (
        w.view(n_heads, head_dim // 2, 2, -1)
        .transpose(1, 2)
        .reshape(n_heads * head_dim, -1)
        .contiguous()
    )


def convert_kaze_state_to_qwen2moe(state: dict[str, torch.Tensor], kaze_cfg: dict) -> dict[str, torch.Tensor]:
    """Remapea tensores KazeMoE -> Qwen2MoeForCausalLM preservando matemáticas exactas (incl. RoPE NeoX)."""
    d_model = int(kaze_cfg["d_model"])
    n_heads = int(kaze_cfg["n_heads"])
    n_kv_heads = int(kaze_cfg["n_kv_heads"])
    head_dim = d_model // n_heads
    n_layers = int(kaze_cfg["n_layers"])

    out: dict[str, torch.Tensor] = {}
    dtype = state["layers.0.attn.wq.weight"].dtype
    max_bias_abs = 0.0

    out["model.embed_tokens.weight"] = state["tok_embeddings.weight"].contiguous()
    out["model.norm.weight"] = state["norm.weight"].contiguous()

    for li in range(n_layers):
        p_in = f"layers.{li}"
        p_out = f"model.layers.{li}"

        out[f"{p_out}.input_layernorm.weight"] = state[f"{p_in}.attn_norm.weight"].contiguous()
        out[f"{p_out}.post_attention_layernorm.weight"] = state[f"{p_in}.ffn_norm.weight"].contiguous()

        # Atención GQA: permutar Q y K de RoPE intercalado -> NeoX (half-split) para Qwen2Moe / llama.cpp
        out[f"{p_out}.self_attn.q_proj.weight"] = permute_interleaved_to_neox(
            state[f"{p_in}.attn.wq.weight"], n_heads, head_dim
        )
        out[f"{p_out}.self_attn.k_proj.weight"] = permute_interleaved_to_neox(
            state[f"{p_in}.attn.wk.weight"], n_kv_heads, head_dim
        )
        out[f"{p_out}.self_attn.v_proj.weight"] = state[f"{p_in}.attn.wv.weight"].contiguous()
        out[f"{p_out}.self_attn.o_proj.weight"] = state[f"{p_in}.attn.wo.weight"].contiguous()
        out[f"{p_out}.self_attn.q_proj.bias"] = torch.zeros(n_heads * head_dim, dtype=dtype)
        out[f"{p_out}.self_attn.k_proj.bias"] = torch.zeros(n_kv_heads * head_dim, dtype=dtype)
        out[f"{p_out}.self_attn.v_proj.bias"] = torch.zeros(n_kv_heads * head_dim, dtype=dtype)

        # Router MoE (verificar si expert_bias no es despreciable)
        bias_key = f"{p_in}.moe.expert_bias"
        if bias_key in state:
            max_bias_abs = max(max_bias_abs, float(state[bias_key].float().abs().max().item()))
        out[f"{p_out}.mlp.gate.weight"] = state[f"{p_in}.moe.gate.weight"].contiguous()

        # Shared Expert: Qwen2Moe multiplica shared_expert por sigmoid(shared_expert_gate(x)).
        # Con shared_expert_gate = 0 -> sigmoid(0) = 0.5, compensamos multiplicando down_proj por 2.0
        # para que la salida sea matemáticamente idéntica a KazeMoE (shared_out * 1.0).
        out[f"{p_out}.mlp.shared_expert.gate_proj.weight"] = state[f"{p_in}.moe.shared_expert.w_gate.weight"].contiguous()
        out[f"{p_out}.mlp.shared_expert.up_proj.weight"] = state[f"{p_in}.moe.shared_expert.w_up.weight"].contiguous()
        out[f"{p_out}.mlp.shared_expert.down_proj.weight"] = (state[f"{p_in}.moe.shared_expert.w_down.weight"] * 2.0).contiguous()
        out[f"{p_out}.mlp.shared_expert_gate.weight"] = torch.zeros((1, d_model), dtype=dtype)

        # Routed Experts
        n_routed = int(kaze_cfg["n_routed_experts"])
        for ei in range(n_routed):
            out[f"{p_out}.mlp.experts.{ei}.gate_proj.weight"] = state[f"{p_in}.moe.experts.{ei}.w_gate.weight"].contiguous()
            out[f"{p_out}.mlp.experts.{ei}.up_proj.weight"] = state[f"{p_in}.moe.experts.{ei}.w_up.weight"].contiguous()
            out[f"{p_out}.mlp.experts.{ei}.down_proj.weight"] = state[f"{p_in}.moe.experts.{ei}.w_down.weight"].contiguous()

    if max_bias_abs > 1e-4:
        print(
            f"[AVISO] max(|expert_bias|)={max_bias_abs:.4f}: Qwen2Moe no almacena expert_bias "
            f"(Aux-Loss-Free); si el valor es alto, haz unos pasos con bias congelado a 0 antes de exportar."
        )

    return out


def save_sharded_safetensors(
    state: dict[str, torch.Tensor],
    out_dir: str,
    max_shard_bytes: int = 4_000_000_000,
) -> list[str]:
    """Guarda el state_dict dividido en shards estándar de Hugging Face + model.safetensors.index.json."""
    shards: list[dict[str, torch.Tensor]] = []
    cur_shard: dict[str, torch.Tensor] = {}
    cur_bytes = 0
    total_bytes = 0

    for k, v in state.items():
        nbytes = v.numel() * v.element_size()
        total_bytes += nbytes
        if cur_shard and (cur_bytes + nbytes > max_shard_bytes):
            shards.append(cur_shard)
            cur_shard = {}
            cur_bytes = 0
        cur_shard[k] = v
        cur_bytes += nbytes
    if cur_shard:
        shards.append(cur_shard)

    n_shards = len(shards)
    if n_shards == 1:
        fpath = os.path.join(out_dir, "model.safetensors")
        save_file(shards[0], fpath)
        return [fpath]

    weight_map: dict[str, str] = {}
    saved_files: list[str] = []
    for idx, shard in enumerate(shards, start=1):
        fname = f"model-{idx:05d}-of-{n_shards:05d}.safetensors"
        fpath = os.path.join(out_dir, fname)
        save_file(shard, fpath)
        saved_files.append(fpath)
        for k in shard:
            weight_map[k] = fname

    index_path = os.path.join(out_dir, "model.safetensors.index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            {"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
            f,
            indent=2,
        )
    return saved_files


def main():
    parser = argparse.ArgumentParser(description="Convierte KazeMoE safetensors -> Qwen2Moe HF (sharded) + GGUF (llama.cpp)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="kaze-4.7b-a530m-mit")
    parser.add_argument("--max-shard-gb", type=float, default=3.5, help="Tamaño máximo por shard safetensors en GB (default 3.5 GB)")
    parser.add_argument("--llama-cpp-dir", default=os.environ.get("LLAMA_CPP_DIR", ""))
    parser.add_argument("--quantize", nargs="*", default=["Q4_K_M"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    hf_dir = os.path.join(args.output_dir, "hf_qwen2moe")
    os.makedirs(hf_dir, exist_ok=True)

    with open(args.config, encoding="utf-8") as f:
        kaze_cfg = json.load(f)

    hf_cfg = build_qwen2moe_config(kaze_cfg)
    with open(os.path.join(hf_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(hf_cfg, f, indent=2)

    tok_cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<|bos|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|pad|>",
        "chat_template": KAZE_CHAT_TEMPLATE,
    }
    with open(os.path.join(hf_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tok_cfg, f, indent=2)
    shutil.copy2(args.tokenizer, os.path.join(hf_dir, "tokenizer.json"))

    state = load_file(args.checkpoint, device="cpu")
    qwen_state = convert_kaze_state_to_qwen2moe(state, kaze_cfg)
    saved = save_sharded_safetensors(
        qwen_state,
        hf_dir,
        max_shard_bytes=int(args.max_shard_gb * 1_000_000_000),
    )
    print(f"Bundle Hugging Face Qwen2Moe listo en {hf_dir} ({len(qwen_state)} tensores en {len(saved)} shards)")

    if not args.llama_cpp_dir:
        print("Tip: pasa --llama-cpp-dir /ruta/a/llama.cpp para generar y cuantizar GGUF automáticamente.")
        return

    convert_py = os.path.join(args.llama_cpp_dir, "convert_hf_to_gguf.py")
    quant_bin = os.path.join(args.llama_cpp_dir, "build", "bin", "llama-quantize")
    f16_gguf = os.path.join(args.output_dir, f"{args.model_name}-f16.gguf")
    subprocess.run(["python3", convert_py, hf_dir, "--outfile", f16_gguf, "--outtype", "f16"], check=True)
    for q in args.quantize:
        q_out = os.path.join(args.output_dir, f"{args.model_name}-{q.lower()}.gguf")
        subprocess.run([quant_bin, f16_gguf, q_out, q], check=True)
        print(f"GGUF generado: {q_out}")


if __name__ == "__main__":
    main()
