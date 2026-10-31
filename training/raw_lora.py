"""Train on existing rendered prompts without applying a second chat template."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from mlx_lm import load
from mlx_lm.lora import CONFIG_DEFAULTS, build_parser, train_model

from hotpotqa_lora import MAX_NEW_TOKENS, encode_prompt, read_jsonl, write_jsonl


def concise_target(tokenizer, completion):
    if len(tokenizer.encode(completion, add_special_tokens=False)) < MAX_NEW_TOKENS:
        return completion
    evidence, answer = completion.rsplit("\nAnswer: ", 1)
    suffix = "\nAnswer: " + answer
    evidence = evidence.removeprefix("Evidence: ")
    excerpt = tokenizer.encode(evidence, add_special_tokens=False)
    while excerpt:
        result = "Evidence excerpt: " + tokenizer.decode(excerpt).rstrip() + "…" + suffix
        if len(tokenizer.encode(result, add_special_tokens=False)) < MAX_NEW_TOKENS:
            return result
        excerpt.pop()
    result = "Answer: " + answer
    if len(tokenizer.encode(result, add_special_tokens=False)) >= MAX_NEW_TOKENS:
        raise ValueError("Gold answer exceeds output budget")
    return result


class EncodedSample:
    def __init__(self, tokens, offset):
        self.tokens, self.offset = tokens, offset

    def __len__(self):
        return len(self.tokens)


class RawDataset:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def process(self, sample):
        return sample.tokens, sample.offset


def main():
    values = vars(build_parser().parse_args())
    for key, default in CONFIG_DEFAULTS.items():
        if values.get(key) is None:
            values[key] = default
    args = SimpleNamespace(**values)
    np.random.seed(args.seed)
    model, tokenizer = load(args.model)
    rows = list(read_jsonl(Path(args.data) / "train.jsonl"))
    for row in rows:
        row["completion"] = concise_target(tokenizer, row["completion"])
    write_jsonl(Path(args.adapter_path).parent / "effective_training_data.jsonl", rows)
    if len(rows) != 5000 or args.iters * args.batch_size != len(rows):
        raise ValueError("This experiment requires exactly one pass over 5,000 examples")
    samples = []
    for row in rows:
        prefix = encode_prompt(tokenizer, row["prompt"])
        target = tokenizer.encode(row["completion"], add_special_tokens=False)
        tokens = prefix + target + [tokenizer.eos_token_id]
        if len(tokens) > args.max_seq_length:
            raise ValueError("Training example exceeds sequence limit; refusing to truncate its answer")
        samples.append(EncodedSample(tokens, len(prefix)))
    root = Path(args.adapter_path).parent
    (root / "training_config.json").write_text(json.dumps({
        **values, "examples": len(samples), "epochs": 1,
        "prompt_format": "Already rendered; encoded once, prompt masked",
        "max_actual_sequence_length": max(map(len, samples)),
    }, indent=2) + "\n")
    print(f"Training {len(samples)} examples, batch {args.batch_size}, {args.iters} steps", flush=True)
    train_model(args, model, RawDataset(samples), RawDataset([]))


if __name__ == "__main__":
    main()
