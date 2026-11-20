import argparse
import itertools
import os
import time
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Prepare uint16 binary shards in /dev/shm for Kaze-1B-A300M")
    parser.add_argument("--dataset", type=str, default="latam-gpt/LatamGPT-Corpus-1.0")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/artifacts/tokenizer.json")
    parser.add_argument("--output-dir", type=str, default="/dev/shm/kaze_data")
    parser.add_argument("--max-docs", type=int, default=200000)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    assert tokenizer.get_vocab_size() <= 65535, "vocab_size must fit in uint16"
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    t0 = time.time()
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    all_tokens = []
    batch = []

    for row in itertools.islice(ds, args.max_docs):
        txt = row.get("text", "")
        if txt and len(txt) > 50:
            batch.append(txt)
        if len(batch) >= args.batch_size:
            for enc in tokenizer.encode_batch(batch):
                all_tokens.extend(enc.ids)
                all_tokens.append(eos_id)
            batch.clear()

    if batch:
        for enc in tokenizer.encode_batch(batch):
            all_tokens.extend(enc.ids)
            all_tokens.append(eos_id)

    arr = np.array(all_tokens, dtype=np.uint16)
    n_val = max(4096, int(len(arr) * args.val_ratio))
    train_path = os.path.join(args.output_dir, "train.bin")
    val_path = os.path.join(args.output_dir, "val.bin")
    arr[:-n_val].tofile(train_path)
    arr[-n_val:].tofile(val_path)

    print(f"Prepared shards in {time.time() - t0:.1f}s")
    print(f"  train.bin: {len(arr) - n_val:,} tokens -> {train_path}")
    print(f"  val.bin:   {n_val:,} tokens -> {val_path}")


if __name__ == "__main__":
    main()
