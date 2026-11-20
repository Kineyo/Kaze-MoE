#!/usr/bin/env python3
"""
Publica el bundle de Kaze-4.7B-A530M-MIT (pesos sharded safetensors + GGUF Q4_K_M + Model Card)
en Hugging Face Hub: shebxyd/Kaze-4.7B-A530M-MIT.
Licencia: MIT.
"""

import argparse
import os
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Sube Kaze-4.7B-A530M-MIT a Hugging Face Hub")
    parser.add_argument("--folder", required=True, help="Carpeta con model-*.safetensors, index.json, config, tokenizer, README.md y *.gguf")
    parser.add_argument("--repo-id", default="shebxyd/Kaze-4.7B-A530M-MIT")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", default="Upload Kaze-4.7B-A530M-MIT (sharded safetensors + Q4_K_M GGUF)")
    args = parser.parse_args()

    if not args.token:
        raise ValueError("Proporciona --token o define la variable de entorno HF_TOKEN.")

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"Subiendo {args.folder} -> https://huggingface.co/{args.repo_id} ...")
    api.upload_folder(
        folder_path=args.folder,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )
    print(f"Publicación completada: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
