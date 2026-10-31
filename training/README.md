# MiniCPM5-1B general QA experiment

## Learn the experiment

- [Interview study guide](INTERVIEW_GUIDE.md): actual method, settings, limitations, and practice questions.
- While the local progress server is running: [animated lesson](http://127.0.0.1:8767/learn) and [live progress](http://127.0.0.1:8767/).
- The lesson has 12 chapters, play/pause controls, captions, and optional narration if a local English device voice is available. It describes results as pending until the final benchmark is measured.

This experiment fine-tunes the Hugging Face `openbmb/MiniCPM5-1B` model with LoRA on exactly 5,000 examples from `hotpotqa/hotpot_qa` (`distractor/train`). Each target includes the dataset's annotated supporting sentences and the gold answer, so the model learns to give a short evidence-based explanation instead of unsupported free-form reasoning.

The before/after benchmark uses **the same fixed 500 questions** from `distractor/validation`, greedy decoding in batches of eight, the same prompt, and HotpotQA's answer exact-match and token-F1 rules. At the user's request, the sample was chosen with seed 137 from the 4,385 baseline questions already completed; the 500 selected IDs are saved in `benchmark_selection.json`. The original validation split contains 7,405 examples. This sample is useful for the paired comparison but is not an official full-split leaderboard result. It also reports results by question type. The dataset is English, Wikipedia-based, and CC BY-SA 4.0; review attribution and ShareAlike terms before distributing the adapter or a merged model. The model is Apache-2.0.

## Run locally on Apple Silicon

Use Python 3.12 and a writable Hugging Face cache. These commands keep downloaded packages and model weights in `/private/tmp`; experiment artifacts remain in `artifacts/model_training/minicpm5_1b_hotpotqa/`.

```sh
export UV_CACHE_DIR=/private/tmp/loveria-uv-cache
export UV_PYTHON_INSTALL_DIR=/private/tmp/loveria-uv-python
export HF_HOME=/private/tmp/loveria-hf-cache

uv run --python 3.12 --with 'mlx-lm[train]' --with datasets \
  python training/hotpotqa_lora.py prepare
uv run --python 3.12 --with 'mlx-lm[train]' --with datasets \
  python training/hotpotqa_lora.py baseline
uv run --python 3.12 --with 'mlx-lm[train]' --with datasets \
  python training/hotpotqa_lora.py train
uv run --python 3.12 --with 'mlx-lm[train]' --with datasets \
  python training/hotpotqa_lora.py after
uv run --python 3.12 --with 'mlx-lm[train]' --with datasets \
  python training/hotpotqa_lora.py report
```

The selected 500-question baseline is complete: **15.2% exact match (76/500), 26.925 token F1**. Training makes one pass over the 5,000 selected rows with prompt tokens masked from the loss, rank-8 LoRA, scale 20, a batch size of 2, 2,500 updates, 16 adapted layers, learning rate 1e-5, seed 137, gradient checkpointing, and a 4,096-token sequence limit. Training checkpoints are saved every 100 updates. It does not upload anything to Hugging Face.

`raw_lora.py` encodes the existing rendered prompt exactly once. MLX's ordinary prompt/completion loader would have applied another chat template. Training targets use annotated evidence and the gold answer; long evidence is shortened to an excerpt so the complete target fits within the benchmark's 96-token generation budget. Gold answers remain intact. This is supervised evidence-and-answer training, not an interactive teacher loop that generates new explanations for individual failures. The effective targets are saved in `effective_training_data.jsonl`.

## Artifacts

- `manifest.json`: model/data IDs, split fingerprints, seed, selected-ID hash and benchmark settings.
- `metrics_baseline.json` and `metrics_after.json`: full benchmark results.
- `predictions_baseline.jsonl` and `predictions_after.jsonl`: per-question predictions for audit.
- `adapters/`: trained LoRA weights.
- `training.log`: fine-tuning output.
- `training_config.json`: actual training settings and maximum sequence length.
- `benchmark_selection.json`: fixed sample IDs and sampling method.
- `effective_training_data.jsonl`: exact 5,000 teaching examples after evidence shortening.
- `report.json`: change in exact match and token F1, including category breakdowns.

This benchmarks the original Hugging Face checkpoint through MLX-LM. It does not validate or replace Loveria's separately converted LiteRT export.

Exact match counts answers that match the gold answer after lowercasing and removing punctuation, articles and extra spaces. Token F1 gives partial credit for word overlap, averaged across questions; yes/no answers follow HotpotQA's strict rule. F1 is not the percentage of fully correct answers. Results on this context-based Wikipedia QA task do not establish an improvement in broad closed-book general knowledge.

Sources: [model](https://huggingface.co/openbmb/MiniCPM5-1B), [dataset](https://huggingface.co/datasets/hotpotqa/hotpot_qa), [official answer scoring](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py), [MLX training](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md).
