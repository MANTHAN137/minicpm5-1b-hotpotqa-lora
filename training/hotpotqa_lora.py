#!/usr/bin/env python3
"""Prepare, benchmark, and LoRA-tune MiniCPM5-1B on HotpotQA."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import string
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


MODEL_ID = "openbmb/MiniCPM5-1B"
DATASET_ID = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"
SEED = 137
TRAIN_ROWS = 5000
BENCHMARK_ROWS = 500
MAX_NEW_TOKENS = 96
EVAL_BATCH_SIZE = 8
SYSTEM_PROMPT = (
    "You are a careful question-answering assistant. Use only the supplied context. "
    "Give a brief explanation grounded in the relevant evidence, then end with exactly "
    "'Answer: <short answer>'. If the context does not support an answer, say so."
)


def run_root(path: str | None) -> Path:
    target = Path(path) if path else Path("artifacts/model_training/minicpm5_1b_hotpotqa")
    target.mkdir(parents=True, exist_ok=True)
    return target.resolve()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def context_text(row: dict) -> str:
    context = row["context"]
    return "\n".join(
        f"[{title}] " + " ".join(sentences)
        for title, sentences in zip(context["title"], context["sentences"])
    )


def evidence_text(row: dict) -> str:
    context = row["context"]
    by_title = defaultdict(list)
    for title, sentences in zip(context["title"], context["sentences"]):
        by_title[title] = sentences
    evidence = []
    seen = set()
    for title, sentence_index in zip(
        row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]
    ):
        sentences = by_title.get(title, [])
        if 0 <= sentence_index < len(sentences):
            item = (title, sentences[sentence_index])
            if item not in seen:
                seen.add(item)
                evidence.append(f"[{title}] {sentences[sentence_index]}")
    return " ".join(evidence)


def user_prompt(question: str, context: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Evidence context:\n{context}\n\n"
        "Answer using the context. Keep the explanation concise and put the final answer "
        "on a line beginning `Answer:`."
    )


def render_prompt(tokenizer, question: str, context: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(question, context)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode_prompt(tokenizer, prompt: str):
    # Match mlx_lm.stream_generate's handling of a string prompt exactly.
    add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
        tokenizer.bos_token
    )
    return tokenizer.encode(prompt, add_special_tokens=add_special_tokens)


def prepare(args):
    from datasets import load_dataset
    from mlx_lm import load

    root = run_root(args.output)
    data_dir = root / "mlx_data"
    print(f"Loading {DATASET_ID} ({DATASET_CONFIG}) from Hugging Face…", flush=True)
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG)
    train = dataset["train"]
    validation = dataset["validation"]
    if len(train) < TRAIN_ROWS:
        raise RuntimeError(f"Expected at least {TRAIN_ROWS} train rows, found {len(train)}")

    rng = random.Random(SEED)
    chosen_indices = sorted(rng.sample(range(len(train)), TRAIN_ROWS))
    chosen = train.select(chosen_indices)

    print(f"Loading model tokenizer from {MODEL_ID}…", flush=True)
    _, tokenizer = load(MODEL_ID)

    training_rows = []
    train_ids = []
    max_prompt_tokens = 0
    for row in chosen:
        prompt = render_prompt(tokenizer, row["question"], context_text(row))
        evidence = evidence_text(row)
        completion = (
            (f"Evidence: {evidence}\n" if evidence else "")
            + f"Answer: {row['answer']}"
        )
        max_prompt_tokens = max(max_prompt_tokens, len(tokenizer.encode(prompt)))
        training_rows.append({"prompt": prompt, "completion": completion})
        train_ids.append(row["id"])

    benchmark_rows = []
    selection_file = root / "benchmark_selection.json"
    if selection_file.exists():
        selection = json.loads(selection_file.read_text())
        selected_ids = set(selection["ids"])
        validation = validation.filter(lambda row: row["id"] in selected_ids)
        if len(validation) != BENCHMARK_ROWS:
            raise ValueError("Saved benchmark selection must contain exactly 500 validation IDs")
    else:
        validation = validation.select(sorted(random.Random(SEED).sample(range(len(validation)), BENCHMARK_ROWS)))
    for row in validation:
        prompt = render_prompt(tokenizer, row["question"], context_text(row))
        benchmark_rows.append(
            {
                "id": row["id"],
                "prompt": prompt,
                "answer": row["answer"],
                "type": row["type"],
                "level": row["level"],
                "supporting_facts": evidence_text(row),
            }
        )

    write_jsonl(data_dir / "train.jsonl", training_rows)
    write_jsonl(root / "benchmark_validation.jsonl", benchmark_rows)
    manifest = {
        "model_id": MODEL_ID,
        "model_parameters": 1_080_632_832,
        "model_license": "Apache-2.0",
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_license": "CC-BY-SA-4.0",
        "dataset_source": "https://huggingface.co/datasets/hotpotqa/hotpot_qa",
        "dataset_fingerprint": {
            split: dataset[split]._fingerprint for split in dataset.keys()
        },
        "train_count": len(training_rows),
        "validation_count": len(benchmark_rows),
        "train_ids_sha256": hashlib.sha256(
            "\n".join(train_ids).encode("utf-8")
        ).hexdigest(),
        "seed": SEED,
        "max_train_prompt_tokens": max_prompt_tokens,
        "training_target": "Evidence copied from annotated supporting facts, then gold answer",
        "benchmark": "Fixed seeded 500-question sample of untouched HotpotQA distractor validation",
        "metrics": ["exact_match", "token_f1"],
        "generation": {
            "temperature": 0.0,
            "max_new_tokens": MAX_NEW_TOKENS,
            "batch_size": EVAL_BATCH_SIZE,
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (root / "DATA_LICENSE.txt").write_text(
        "HotpotQA dataset by its authors, accessed via hotpotqa/hotpot_qa on Hugging Face.\n"
        "Dataset license: Creative Commons Attribution-ShareAlike 4.0 International.\n"
        "Dataset card: https://huggingface.co/datasets/hotpotqa/hotpot_qa\n"
        "The LoRA adapter is an experimental derivative trained on this data. Review the\n"
        "dataset's attribution and ShareAlike terms before distributing the adapter or a\n"
        "merged model. This experiment does not upload either artifact to Hugging Face.\n",
        encoding="utf-8",
    )
    print(
        f"Prepared {len(training_rows)} train rows and {len(benchmark_rows)} validation rows; "
        f"largest training prompt is {max_prompt_tokens} tokens.",
        flush=True,
    )
    print(f"Artifacts: {root}", flush=True)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_answer(text: str) -> str:
    matches = re.findall(r"(?:^|\n)\s*answer\s*:\s*(.*)", text, re.IGNORECASE)
    nonempty_matches = [match.strip() for match in matches if match.strip()]
    if nonempty_matches:
        return nonempty_matches[-1].splitlines()[0].strip()
    if matches:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def token_f1(prediction: str, reference: str) -> float:
    # Match HotpotQA's special handling of yes/no/noanswer answers.
    p, g = normalize_answer(prediction), normalize_answer(reference)
    if p != g and (p in {"yes", "no", "noanswer"} or g in {"yes", "no", "noanswer"}):
        return 0.0
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(reference).split()
    if not pred or not gold:
        return 0.0
    counts = defaultdict(int)
    for token in pred:
        counts[token] += 1
    overlap = 0
    for token in gold:
        if counts[token] > 0:
            overlap += 1
            counts[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def score_records(records):
    n = len(records)
    if not n:
        raise RuntimeError("No benchmark predictions were produced")

    def group_score(rows):
        count = len(rows)
        return {
            "count": count,
            "exact_match": 100 * sum(r["exact_match"] for r in rows) / count,
            "token_f1": 100 * sum(r["token_f1"] for r in rows) / count,
        }

    by_type = defaultdict(list)
    by_level = defaultdict(list)
    for row in records:
        by_type[row["type"]].append(row)
        by_level[row["level"]].append(row)
    return {
        "count": n,
        "exact_match": 100 * sum(r["exact_match"] for r in records) / n,
        "token_f1": 100 * sum(r["token_f1"] for r in records) / n,
        "unparsed_answers": sum(r["unparsed"] for r in records),
        "by_type": {key: group_score(value) for key, value in sorted(by_type.items())},
        "by_level": {key: group_score(value) for key, value in sorted(by_level.items())},
    }


def evaluate(args):
    from mlx_lm import load
    from mlx_lm.generate import batch_generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    root = run_root(args.output)
    benchmark_path = root / "benchmark_validation.jsonl"
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Run the prepare phase first: {benchmark_path}")
    adapter = str(root / "adapters") if args.after else None
    if adapter and not (Path(adapter) / "adapters.safetensors").exists():
        raise FileNotFoundError(f"Trained adapter not found: {adapter}")
    label = "after" if args.after else "baseline"
    prediction_path = root / f"predictions_{label}.jsonl"
    existing = []
    if prediction_path.exists() and not args.overwrite:
        existing = list(read_jsonl(prediction_path))
    elif prediction_path.exists() and args.overwrite:
        prediction_path.unlink()

    print(
        f"Loading {MODEL_ID} ({'with LoRA adapter' if adapter else 'untuned'})…",
        flush=True,
    )
    model, tokenizer = load(MODEL_ID, adapter_path=adapter)
    selected_ids = {row["id"] for row in read_jsonl(benchmark_path)}
    existing = [record for record in existing if record["id"] in selected_ids]
    records = existing
    completed_ids = {record["id"] for record in existing}
    start = time.perf_counter()
    rows = list(read_jsonl(benchmark_path))
    remaining = [row for row in rows if row["id"] not in completed_ids]
    mode = "a" if existing else "w"
    print(
        f"Resuming {label} benchmark at {len(completed_ids)}/{len(rows)} examples.",
        flush=True,
    )
    with prediction_path.open(mode, encoding="utf-8") as stream:
        for offset in range(0, len(remaining), EVAL_BATCH_SIZE):
            batch = remaining[offset : offset + EVAL_BATCH_SIZE]
            completions = batch_generate(
                model,
                tokenizer,
                [encode_prompt(tokenizer, row["prompt"]) for row in batch],
                max_tokens=MAX_NEW_TOKENS,
                sampler=make_sampler(temp=0.0),
                completion_batch_size=EVAL_BATCH_SIZE,
                prefill_batch_size=EVAL_BATCH_SIZE,
                verbose=False,
            ).texts
            for row, completion in zip(batch, completions):
                prediction = extract_answer(completion)
                record = {
                    "id": row["id"],
                    "prediction": prediction,
                    "reference": row["answer"],
                    "type": row["type"],
                    "level": row["level"],
                    "exact_match": normalize_answer(prediction)
                    == normalize_answer(row["answer"]),
                    "token_f1": token_f1(prediction, row["answer"]),
                    "unparsed": "answer:" not in completion.lower() or not prediction,
                    "raw_completion": completion,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
                completed_ids.add(row["id"])
            stream.flush()
            index = len(completed_ids)
            if index // 100 > (index - len(batch)) // 100 or index == len(rows):
                elapsed = time.perf_counter() - start
                current = score_records(records)
                print(
                    f"{label}: {index}/{len(rows)} | EM {current['exact_match']:.2f}% | "
                    f"F1 {current['token_f1']:.2f}% | {elapsed / 60:.1f} min",
                    flush=True,
                )
            if index % 64 == 0:
                mx.clear_cache()

    metrics = score_records(records)
    metrics.update(
        {
            "model_id": MODEL_ID,
            "adapter": "LoRA" if adapter else None,
            "split": "hotpotqa/distractor/validation",
            "prediction_file": str(prediction_path),
            "wall_time_seconds": round(time.perf_counter() - start, 2),
            "generation": {
                "temperature": 0.0,
                "max_new_tokens": MAX_NEW_TOKENS,
                "batch_size": EVAL_BATCH_SIZE,
            },
        }
    )
    (root / f"metrics_{label}.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)


def train(args):
    root = run_root(args.output)
    train_file = root / "mlx_data" / "train.jsonl"
    if not train_file.exists():
        raise FileNotFoundError(f"Run the prepare phase first: {train_file}")
    before = root / "metrics_baseline.json"
    if not before.exists():
        raise FileNotFoundError("Run the baseline benchmark before training")
    command = [
        sys.executable,
        str(Path(__file__).with_name("raw_lora.py")),
        "--model",
        MODEL_ID,
        "--train",
        "--data",
        str(train_file.parent),
        "--adapter-path",
        str(root / "adapters"),
        "--fine-tune-type",
        "lora",
        "--mask-prompt",
        "--batch-size",
        "2",
        "--iters",
        "2500",
        "--learning-rate",
        "0.00001",
        "--num-layers",
        "16",
        "--max-seq-length",
        "4096",
        "--grad-checkpoint",
        "--seed",
        str(SEED),
    ]
    log_path = root / "training.log"
    print("Starting one pass over 5,000 examples with LoRA…", flush=True)
    print(" ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    if not (root / "adapters" / "adapters.safetensors").exists():
        raise RuntimeError("Training finished but the LoRA adapter file is missing")
    print(f"Training complete. Log: {log_path}", flush=True)


def report(args):
    root = run_root(args.output)
    before = json.loads((root / "metrics_baseline.json").read_text(encoding="utf-8"))
    after = json.loads((root / "metrics_after.json").read_text(encoding="utf-8"))
    baseline_ids = {r["id"] for r in read_jsonl(root / "predictions_baseline.jsonl")}
    after_ids = {r["id"] for r in read_jsonl(root / "predictions_after.jsonl")}
    if baseline_ids != after_ids or len(baseline_ids) != BENCHMARK_ROWS:
        raise ValueError("Before/after results must cover the identical 500 question IDs")
    report = {
        "model_id": MODEL_ID,
        "training_examples": TRAIN_ROWS,
        "benchmark": "Fixed seeded 500-question HotpotQA distractor validation sample",
        "examples_scored": before["count"],
        "baseline_exact_match": before["exact_match"],
        "after_exact_match": after["exact_match"],
        "exact_match_change_points": after["exact_match"] - before["exact_match"],
        "baseline_token_f1": before["token_f1"],
        "after_token_f1": after["token_f1"],
        "token_f1_change_points": after["token_f1"] - before["token_f1"],
        "by_type": {},
        "by_level": {},
        "limitations": [
            "HotpotQA is multi-hop Wikipedia QA with supplied evidence context, not closed-book factual recall.",
            "The local full-precision/HF-derived checkpoint is benchmarked; Loveria's LiteRT export is a separate runtime that still needs native validation.",
            "The adapter is experimental and trained on CC BY-SA 4.0 data; review attribution and ShareAlike terms before distribution.",
        ],
    }
    for section in ("by_type", "by_level"):
        for key in sorted(set(before[section]) | set(after[section])):
            b = before[section].get(key, {})
            a = after[section].get(key, {})
            report[section][key] = {
                "count": a.get("count", b.get("count", 0)),
                "exact_match_change_points": a.get("exact_match", 0)
                - b.get("exact_match", 0),
                "token_f1_change_points": a.get("token_f1", 0) - b.get("token_f1", 0),
            }
    target = root / "report.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    config = json.loads((root / "training_config.json").read_text())
    selection = json.loads((root / "benchmark_selection.json").read_text())
    documentation = f"""# MiniCPM5-1B training experiment

## Results

Baseline means the original model's score before this experiment's training.
Both evaluations use the identical {before['count']} saved question IDs.

| Answer metric | Before training | After training | Change (percentage points) |
| --- | ---: | ---: | ---: |
| Exact match | {before['exact_match']:.3f}% | {after['exact_match']:.3f}% | {report['exact_match_change_points']:+.3f} |
| Token F1 | {before['token_f1']:.3f}/100 | {after['token_f1']:.3f}/100 | {report['token_f1_change_points']:+.3f} |

Exact match counts fully matching normalized answers. F1 measures word overlap,
giving partial credit; it is not the percentage of fully correct answers.
HotpotQA's official yes/no/noanswer scoring rules apply.
Unparsed or empty answer outputs: before {before['unparsed_answers']}, after {after['unparsed_answers']}.

## Model and data

- Model: `{MODEL_ID}`, 1,080,632,832 parameters, Apache-2.0.
- Source: `{DATASET_ID}`, `distractor` configuration, English Wikipedia questions with evidence context.
- Training: exactly 5,000 examples sampled from the training split, seed {SEED}.
- Evaluation: 500 questions randomly sampled from {selection['candidate_count']} already completed baseline questions, seed {selection['seed']}.
- The original validation split has 7,405 questions. The full run was stopped at the user's request to shorten the experiment.
- Sampling used question IDs without selecting by answer correctness. IDs are in `benchmark_selection.json`.
- A 500-question sample is a limited estimate. This is not an official full-split leaderboard score or proof of broad closed-book general-knowledge improvement.

## Training method

Local MLX on the Apple Silicon machine. Supervised targets contain annotated
supporting evidence and the correct answer. Long explanations are shortened to
evidence excerpts so the complete target fits the 96-token evaluation budget;
gold answers are preserved. This is supervised correction, not an interactive
teacher that diagnoses each failure or generates hidden thinking traces.

Raw rendered prompts are encoded once and masked from training loss. A custom
loader prevents a second chat template from being applied. Training refuses to
truncate examples beyond the sequence limit.

- LoRA rank {config['lora_parameters']['rank']}, scale {config['lora_parameters']['scale']}, {config['num_layers']} adapted layers.
- {config['iters']} updates × batch size {config['batch_size']} = 5,000 examples, one pass.
- Learning rate {config['learning_rate']}; seed {config['seed']}; gradient checkpointing enabled.
- Maximum sequence limit {config['max_seq_length']}; actual largest sequence {config['max_actual_sequence_length']} tokens.
- Adapter checkpoints every {config['save_every']} updates. Actual settings: `training_config.json`.

## Evaluation protocol and evidence

Same prompt, greedy decoding (temperature 0), output limit 96 tokens, and batch
size 8 before and after. Reference answers are used for scoring, never supplied
in evaluation prompts. Per-question outputs are saved in
`predictions_baseline.jsonl` and `predictions_after.jsonl`. Aggregate metrics and
question-type breakdowns are in `metrics_baseline.json`, `metrics_after.json`,
and `report.json`. The fixed benchmark is `benchmark_validation.jsonl`.
Exact training targets are in `effective_training_data.jsonl`; trained weights
are `adapters/adapters.safetensors`; runtime progress is in `training.log`.

The initial short training attempt was interrupted before restarting with
shortened evidence targets. Final training starts from the original model.
The earlier partial full-validation predictions remain archived separately.

## Reproduction and limitations

See `training/README.md` for environment variables and phase commands, and
`manifest.json` for fingerprints and the training-ID hash. Keep
`benchmark_selection.json` to reproduce the same evaluation sample.

The benchmark supplies evidence passages. It does not measure all forms of
general knowledge, reasoning, or conversational ability. Evidence selection
and joint supporting-fact metrics are not scored. Results cover one seed and
one training run. No Android/LiteRT export is validated or installed by this
experiment. No model or adapter is uploaded.

Dataset: CC BY-SA 4.0; model: Apache-2.0. Attribution and any applicable
ShareAlike requirements should be reviewed before redistribution.

## Sources

- [Model](https://huggingface.co/openbmb/MiniCPM5-1B)
- [Dataset](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
- [Official scoring](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)
- [MLX training](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
"""
    (root / "experiment_report.md").write_text(documentation, encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved: {target}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "baseline", "train", "after", "report"])
    parser.add_argument("--output", help="Experiment artifact directory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args)
    elif args.phase == "baseline":
        evaluate(argparse.Namespace(output=args.output, after=False, overwrite=args.overwrite))
    elif args.phase == "train":
        train(args)
    elif args.phase == "after":
        evaluate(argparse.Namespace(output=args.output, after=True, overwrite=args.overwrite))
    else:
        report(args)


if __name__ == "__main__":
    main()
