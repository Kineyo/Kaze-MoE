import argparse
import json
import math
import os
import random
import sys
import time
import torch
import torch.nn.functional as F
from datasets import load_dataset
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.kaze_moe import KazeConfig, KazeMoEModel


KAZE_IDENTITY = [
    ("Hola", "¡Hola! Soy Kaze, un asistente de inteligencia artificial basado en arquitectura Mixture-of-Experts. ¿En qué puedo ayudarte hoy?"),
    ("¿Quién eres?", "Soy Kaze, un modelo de lenguaje ligero desarrollado por Kineyo con arquitectura Sparse Mixture-of-Experts, diseñado para conversar y ayudar de forma rápida y eficiente."),
    ("¿Qué puedes hacer?", "Puedo mantener conversaciones en español, responder preguntas generales, resumir ideas, redactar textos y ayudarte a resolver dudas técnicas o cotidianas."),
    ("Gracias", "¡De nada! Si necesitas algo más, aquí estoy para ayudarte."),
]


def load_spanish_conversations() -> list[list[tuple[str, str]]]:
    convs: list[list[tuple[str, str]]] = []
    for _ in range(15):
        for u, a in KAZE_IDENTITY:
            convs.append([("user", u), ("assistant", a)])

    # OpenAssistant/oasst1 (multi-turn Spanish)
    oasst = load_dataset("OpenAssistant/oasst1", split="train")
    es_msgs = [m for m in oasst if m.get("lang") == "es" and not m.get("deleted", False) and m.get("text")]
    children: dict[str, list[dict]] = {}
    for m in es_msgs:
        if m.get("parent_id"):
            children.setdefault(m["parent_id"], []).append(m)

    for m in es_msgs:
        if m.get("parent_id") is None and m.get("role") == "prompter":
            conv = []
            curr = m
            while curr is not None:
                role = "user" if curr["role"] == "prompter" else "assistant"
                conv.append((role, curr["text"].strip()))
                ch = children.get(curr["message_id"], [])
                if not ch:
                    break
                ch.sort(key=lambda x: (x.get("rank") if x.get("rank") is not None else 99))
                curr = ch[0]
            if len(conv) >= 2 and conv[-1][0] == "assistant":
                convs.append(conv)

    # bertin-project/alpaca-spanish
    alpaca = load_dataset("bertin-project/alpaca-spanish", split="train")
    for row in alpaca:
        inst = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        out = (row.get("output") or "").strip()
        if inst and out and len(out) > 10:
            user_msg = f"{inst}\n{inp}" if inp else inst
            convs.append([("user", user_msg), ("assistant", out)])

    return convs


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning (SFT) conversacional para Kaze-MoE")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints/kaze-instruct")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--min-lr", type=float, default=1.5e-5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    with open(args.config) as f:
        cfg = KazeConfig(**json.load(f))

    model = KazeMoEModel(cfg).to(device=device, dtype=dtype)
    state = load_file(args.checkpoint)
    model.load_state_dict(state, strict=False)
    model.lm_head.weight = model.tok_embeddings.weight

    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    pad_id = tokenizer.token_to_id("<|pad|>")
    user_id = tokenizer.token_to_id("<|user|>")
    asst_id = tokenizer.token_to_id("<|assistant|>")
    nl_ids = tokenizer.encode("\n").ids

    convs = load_spanish_conversations()
    samples = []
    for conv in convs:
        input_ids, labels = [], []
        for role, msg in conv:
            tag_id = user_id if role == "user" else asst_id
            msg_ids = tokenizer.encode(msg).ids
            if role == "user":
                chunk = [tag_id] + nl_ids + msg_ids + nl_ids
                input_ids.extend(chunk)
                labels.extend([-100] * len(chunk))
            else:
                prefix = [tag_id] + nl_ids
                resp = msg_ids + [eos_id] + nl_ids
                input_ids.extend(prefix + resp)
                labels.extend([-100] * len(prefix) + resp)
        if len(input_ids) >= 16:
            input_ids = input_ids[: args.seq_len + 1]
            labels = labels[: args.seq_len + 1]
            if any(l != -100 for l in labels[1:]):
                samples.append((input_ids, labels))

    print(f"SFT samples: {len(samples):,} | steps={args.steps} | device={device}")

    decay_p = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_p = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay_p, "weight_decay": 0.05}, {"params": nodecay_p, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=(device == "cuda"),
    )

    model.train()
    t0 = time.time()
    for step in range(args.steps):
        if step < 40:
            lr = args.lr * (step + 1) / 40
        else:
            prog = (step - 40) / max(1, args.steps - 40)
            lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * prog))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        batch = random.sample(samples, args.batch_size)
        x_list, y_list = [], []
        for inp, lab in batch:
            x, y = inp[:-1], lab[1:]
            pad_len = args.seq_len - len(x)
            if pad_len > 0:
                x = x + [pad_id] * pad_len
                y = y + [-100] * pad_len
            x_list.append(x)
            y_list.append(y)

        xb = torch.tensor(x_list, dtype=torch.long, device=device)
        yb = torch.tensor(y_list, dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_expert_biases()

        if step % 100 == 0 or step == args.steps - 1:
            print(f"[SFT] step {step:04d}/{args.steps} | sft_loss {loss.item():.4f} | lr {lr:.2e}")

    out_ckpt = os.path.join(args.output_dir, "kaze_instruct.safetensors")
    state_dict = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    save_file(state_dict, out_ckpt)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)
    print(f"Saved SFT checkpoint to {out_ckpt} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
