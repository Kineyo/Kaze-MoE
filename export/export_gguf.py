import argparse
import json
import os
import shutil
import subprocess
from safetensors.torch import load_file, save_file


def build_hf_config(kaze_cfg: dict) -> dict:
    return {
        "architectures": ["KazeMoEForCausalLM"],
        "model_type": "kaze_moe",
        "vocab_size": kaze_cfg["vocab_size"],
        "hidden_size": kaze_cfg["d_model"],
        "num_hidden_layers": kaze_cfg["n_layers"],
        "num_attention_heads": kaze_cfg["n_heads"],
        "num_key_value_heads": kaze_cfg["n_kv_heads"],
        "intermediate_size": kaze_cfg["d_ff"],
        "num_routed_experts": kaze_cfg["n_routed_experts"],
        "num_shared_experts": kaze_cfg["n_shared_experts"],
        "num_experts_per_tok": kaze_cfg["top_k"],
        "max_position_embeddings": kaze_cfg["max_seq_len"],
        "rope_theta": kaze_cfg["rope_theta"],
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
    }


def main():
    parser = argparse.ArgumentParser(description="Export Kaze-1B-A300M .safetensors checkpoint to GGUF")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="export/out")
    parser.add_argument("--llama-cpp-dir", type=str, default=os.environ.get("LLAMA_CPP_DIR", ""))
    parser.add_argument("--quantize", nargs="*", default=["Q4_K_M", "IQ4_XS", "Q8_0"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    hf_dir = os.path.join(args.output_dir, "hf_bundle")
    os.makedirs(hf_dir, exist_ok=True)

    with open(args.config) as f:
        kaze_cfg = json.load(f)

    hf_cfg = build_hf_config(kaze_cfg)
    with open(os.path.join(hf_dir, "config.json"), "w") as f:
        json.dump(hf_cfg, f, indent=2)

    tensors = load_file(args.checkpoint)
    save_file(tensors, os.path.join(hf_dir, "model.safetensors"))
    shutil.copy2(args.tokenizer, os.path.join(hf_dir, "tokenizer.json"))
    print(f"Prepared safetensors bundle in {hf_dir}")

    if not args.llama_cpp_dir:
        print("LLAMA_CPP_DIR not set. Pass --llama-cpp-dir /path/to/llama.cpp to run convert_hf_to_gguf.py & llama-quantize.")
        return

    convert_script = os.path.join(args.llama_cpp_dir, "convert_hf_to_gguf.py")
    quant_bin = os.path.join(args.llama_cpp_dir, "build", "bin", "llama-quantize")
    f16_gguf = os.path.join(args.output_dir, "Kaze-1B-A300M-F16.gguf")

    subprocess.run(["python3", convert_script, hf_dir, "--outfile", f16_gguf, "--outtype", "f16"], check=True)
    for q in args.quantize:
        q_out = os.path.join(args.output_dir, f"Kaze-1B-A300M-{q}.gguf")
        subprocess.run([quant_bin, f16_gguf, q_out, q], check=True)
        print(f"Generated {q_out}")


if __name__ == "__main__":
    main()
