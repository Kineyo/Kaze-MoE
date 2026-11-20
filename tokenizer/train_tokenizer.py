import argparse
import itertools
import os
import time
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers


SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|bos|>",
    "<|pad|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
]


def text_iterator(dataset_name: str, split: str, max_docs: int, text_column: str = "text"):
    ds = load_dataset(dataset_name, split=split, streaming=True)
    for row in itertools.islice(ds, max_docs):
        txt = row.get(text_column, "")
        if txt and len(txt) > 50:
            yield txt


def main():
    parser = argparse.ArgumentParser(description="Train Byte-Level BPE tokenizer for Kaze-1B-A300M")
    parser.add_argument("--dataset", type=str, default="latam-gpt/LatamGPT-Corpus-1.0")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--max-docs", type=int, default=50000)
    parser.add_argument("--output-dir", type=str, default="tokenizer/artifacts")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "tokenizer.json")

    t0 = time.time()
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train_from_iterator(
        text_iterator(args.dataset, args.split, args.max_docs),
        trainer=trainer,
        length=args.max_docs,
    )
    tokenizer.save(out_path)
    print(f"Saved tokenizer ({tokenizer.get_vocab_size()} tokens) to {out_path} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
