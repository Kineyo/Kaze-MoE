import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.kaze_moe import KazeConfig, KazeMoEModel


def get_batch(data: np.memmap, batch_size: int, seq_len: int, device: str):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + seq_len].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def get_wsd_lr(step: int, max_steps: int, warmup_steps: int, decay_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    stable_end = max_steps - decay_steps
    if step < stable_end:
        return max_lr
    decay_ratio = (step - stable_end) / max(1, decay_steps)
    return max_lr - (max_lr - min_lr) * min(1.0, decay_ratio)


@torch.no_grad()
def evaluate(model: KazeMoEModel, val_data: np.memmap, batch_size: int, seq_len: int, device: str, n_batches: int = 10):
    model.eval()
    losses = []
    for _ in range(n_batches):
        xb, yb = get_batch(val_data, batch_size, seq_len, device)
        _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser(description="Pre-train Kaze-1B-A300M with WSD scheduler in bfloat16")
    parser.add_argument("--data-dir", type=str, default="/dev/shm/kaze_data")
    parser.add_argument("--output-dir", type=str, default="checkpoints/kaze-1b-a300m")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--decay-steps", type=int, default=5000)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    cfg = KazeConfig(use_checkpoint=args.grad_checkpoint)
    raw_model = KazeMoEModel(cfg).to(device=device, dtype=dtype)
    total_p, active_p = raw_model.count_parameters()
    print(f"Kaze-1B-A300M | total={total_p / 1e9:.3f}B | active={active_p / 1e6:.1f}M | device={device} ({dtype})")

    model = torch.compile(raw_model) if args.compile else raw_model

    train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")

    decay_params = [p for p in raw_model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in raw_model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=args.max_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=(device == "cuda"),
    )

    tokens_per_step = args.micro_batch * args.grad_accum * args.seq_len
    model.train()
    for step in range(args.max_steps):
        t0 = time.time()
        lr = get_wsd_lr(step, args.max_steps, args.warmup_steps, args.decay_steps, args.max_lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            xb, yb = get_batch(train_data, args.micro_batch, args.seq_len, device)
            _, loss = model(xb, yb)
            (loss / args.grad_accum).backward()
            accum_loss += loss.item() / args.grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()
        raw_model.update_expert_biases()
        dt = time.time() - t0

        if step % 10 == 0 or step == args.max_steps - 1:
            msg = f"step {step:05d}/{args.max_steps} | loss {accum_loss:.4f} | lr {lr:.2e} | grad {grad_norm:.2f} | {tokens_per_step / dt:,.0f} tok/s"
            if step % args.eval_interval == 0 or step == args.max_steps - 1:
                val_loss = evaluate(raw_model, val_data, args.micro_batch, args.seq_len, device)
                msg += f" | val_loss {val_loss:.4f}"
            print(msg)

    ckpt_path = os.path.join(args.output_dir, "kaze_1b_a300m.safetensors")
    state_dict = {k: v.detach().cpu().contiguous() for k, v in raw_model.state_dict().items() if k != "lm_head.weight"}
    save_file(state_dict, ckpt_path)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
