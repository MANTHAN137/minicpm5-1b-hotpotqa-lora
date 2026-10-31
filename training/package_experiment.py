"""Build a standalone publication from the completed experiment's recorded files."""
import csv
import importlib.util
import json
from pathlib import Path
import random
import re
import shutil

PROJECT = Path(__file__).resolve().parents[1]
RUN = PROJECT / "artifacts/model_training/minicpm5_1b_hotpotqa"
DEST = PROJECT / "artifacts/publication/minicpm5-1b-hotpotqa-lora"
REPO = "https://github.com/MANTHAN137/minicpm5-1b-hotpotqa-lora"


def main():
    def read(name):
        return json.loads((RUN / name).read_text())

    spec = importlib.util.spec_from_file_location("progress", PROJECT / "training/progress_dashboard.py")
    ui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui)
    examples = ui.examples()
    assert len(examples) == 500 and len({r["id"] for r in examples}) == 500
    b, a, paired = read("metrics_baseline.json"), read("metrics_after.json"), read("paired_analysis.json")
    config, audit = read("training_config.json"), read("dataset_split_audit.json")
    assert b["count"] == a["count"] == 500
    assert not audit["overlapping_question_ids"] and not audit["overlapping_normalized_question_texts"]
    assert config["iters"] * config["batch_size"] == 5000
    diffs = [(int(r["after"]["exact_match"]) - int(r["before"]["exact_match"]), r["after"]["token_f1"] - r["before"]["token_f1"]) for r in examples]
    rng = random.Random(137)
    em, f1 = [], []
    for _ in range(5000):
        sample = rng.choices(diffs, k=500)
        em.append(sum(x[0] for x in sample) / 5)
        f1.append(sum(x[1] for x in sample) / 5)
    em.sort(); f1.sort()
    uncertainty = {"method": "Paired percentile bootstrap over the 500 fixed evaluation questions", "resamples": 5000, "seed": 137,
                   "confidence_level": 0.95, "exact_match_change_points_interval": [em[124], em[4874]],
                   "token_f1_change_points_interval": [f1[124], f1[4874]],
                   "scope": "Question resampling only; does not capture training-seed variability, dataset shift or pretraining contamination"}
    (RUN / "uncertainty.json").write_text(json.dumps(uncertainty, indent=2) + "\n")

    # Use standard plotting tools for an exportable research figure.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (RUN / "figures").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, title, before, after in [(axes[0], "Answer exact match", b["exact_match"], a["exact_match"]), (axes[1], "Answer token F1", b["token_f1"], a["token_f1"])]:
        bars = ax.bar(["Before", "After"], [before, after], color=["#7b8798", "#2676b8"], width=.55)
        ax.set(ylim=(0, 100), ylabel="Score / 100", title=title)
        ax.bar_label(bars, labels=[f"{before:.2f}", f"{after:.2f}"], padding=4)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("MiniCPM5-1B · same 500 held-out HotpotQA questions")
    fig.savefig(RUN / "figures/before-after.png", dpi=180)
    fig.savefig(RUN / "figures/before-after.svg")
    plt.close(fig)

    original = (RUN / "experiment_report.md").read_text()
    original = original.split("\n## Interpretation and remaining headroom")[0]
    discussion = f"""
## Interpretation and remaining headroom

The experiment demonstrates a task-specific gain, not a measurement of general intelligence or a demonstrated model ceiling. Exact-match answers rose from **76/500 to 162/500**, a gain of **17.2 percentage points**. That is about 2.13 times the original exact-match count, but **338/500 responses still fail exact match**. Some semantic answers may be penalized by strict string matching; F1 gives partial overlap credit.

| Paired outcome | Questions |
| --- | ---: |
| Correct in both runs | {paired['both_correct']} |
| Wrong before, correct after | {paired['newly_correct']} |
| Correct before, wrong after | {paired['regressed']} |
| Wrong in both runs | {paired['both_incorrect']} |

The 120 gains and 34 regressions show that adaptation does not help every example. Bridge questions gained 19.51 EM points; comparison questions gained 7.37 points. These are descriptive breakdowns, not separate tuning targets or causal explanations.

Empty parsed answers increased from **1 to 18**. The broader missing-marker-or-empty flag decreased from 51 to 36, so these two diagnostics should not be conflated. Inspect the verbatim responses in the UI: some explanations consume the output budget before a complete final answer appears. Output length, target format and actual reasoning failures need controlled diagnosis; the current run cannot attribute the gain or remaining errors to any one factor.

### Is 32.4% the model's limit?

We do not know. One epoch, one seed and one hyperparameter configuration do not estimate an architecture's ultimate ceiling. There is no learning-curve sweep or controlled comparison against full fine-tuning, alternative datasets, or larger models. Lower training loss proves a better fit to the teaching targets, not further held-out improvement. More data, epochs or adapter capacity could help, plateau, overfit, or regress.

The initial 15.2% is performance under this specific prompt, evidence context, output budget and strict scorer. HotpotQA asks for evidence use across multiple Wikipedia passages, with distractors. It is not an IQ test or a broad general-knowledge score. See the [HotpotQA paper](https://arxiv.org/abs/1809.09600).

### How much confidence should we place in the measured gain?

A paired bootstrap resampling these 500 questions 5,000 times gives an approximate 95% interval of **{em[124]:.2f} to {em[4874]:.2f} percentage points** for EM change and **{f1[124]:.2f} to {f1[4874]:.2f} points** for F1 change. This describes question-sampling variability on this explored sample. It does not account for training-seed variation, sample-selection bias, public pretraining contamination, or generalization to other tasks. It is not evidence that the model is ready for broad deployment.

### Proposed next experiments — not performed in this run

1. Create a separate development set outside the 5,000 training examples and this explored benchmark. Reserve a fresh final evaluation set before choosing further settings.
2. Diagnose format and truncation on development data: compare output budgets and concise answer-first versus evidence-first targets. If prompts or decoding change, compare base and tuned models under the same new protocol; keep this report's original results intact.
3. Audit a structured sample of factual, evidence-selection, comparison and formatting errors. Improve teaching-target quality based on development and training data.
4. Measure development learning curves at fractions of an epoch and across additional epochs. Stop when development quality stops improving; low training loss alone is not the selection rule.
5. Run a small, recorded learning-rate/rank/layer sweep and repeat promising settings across seeds. Increase data quantity only with quality and leakage checks.
6. If needed, compare a larger base model under matched prompts, data and scoring. This distinguishes adaptation effects from model-size effects without assuming either will solve the task.

After displaying the 500 questions and their outputs, treat them as an **explored benchmark**. Do not repeatedly optimize against their answers and continue calling them an untouched final exam. No additional training has been started by this discussion.

## Data-split verification

The actual training-ID hash matches the preparation manifest. Prepared and effective training prompts match. The 5,000 training examples and 500 benchmark questions have **zero overlapping IDs and zero overlapping normalized question texts**. This does not rule out shared facts/passages or the original model's pretraining exposure. See `dataset_split_audit.json`.

## Figures and publication contents

![Before and after scores on the fixed 500 questions](figures/before-after.png)

Published materials include source code, the fixed benchmark IDs, configuration, aggregate and paired metrics, a score table, the interview guide, and verbatim before/after comparisons. The local base-model cache, 5,000 training texts, intermediate checkpoints and final adapter weights are not included in this repository. The code can train another adapter locally. The static results page does not run inference.

The 500-question comparison collection includes HotpotQA excerpts and model-generated outputs. Dataset-derived material is attributed in `DATA_NOTICE.md` and supplied under CC BY-SA 4.0; code is licensed separately. Generated responses may contain errors and do not represent verified supporting evidence.
"""
    summary = read("training_data_summary.json")
    samples_text = "\n".join(f"- **{row['type']} / {row['level']}**: {row['question']} **Gold answer:** {row['answer']}." for row in summary['samples'])
    dataset_section = f"""
## Actual training data and measured task categories

The selected 5,000 training examples contain **3,966 bridge questions and 1,034 comparison questions**. Difficulty labels are 1,010 easy, 3,127 medium and 863 hard. Training targets combine supporting evidence and the gold answer; the full input includes the supplied context passages. The following are real training examples, not held-out evaluation examples:

{samples_text}

Some shortened teaching excerpts omit useful comparison details. This is a target-quality limitation to investigate rather than evidence that further learning is impossible. The exact targets for these examples are included in `training_data_summary.json` and shown on the public page.

| Question type | Evaluation questions | Before EM | After EM | EM change | Before F1 | After F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bridge | 405 | 14.32% | 33.83% | +19.51 points | 26.34 | 50.59 |
| Comparison | 95 | 18.95% | 26.32% | +7.37 points | 29.43 | 39.92 |

Bridge questions connect facts across passages; comparison questions compare entities or quantities. These are task categories. No separate subject-domain evaluation establishes improvement in medicine, coding, mathematics or other specialties.
"""
    report = original + discussion + dataset_section
    (RUN / "experiment_report.md").write_text(report)

    DEST.mkdir(parents=True, exist_ok=True)
    public_run = DEST / "artifacts/model_training/minicpm5_1b_hotpotqa"
    public_run.mkdir(parents=True, exist_ok=True)
    (DEST / "training").mkdir(exist_ok=True)
    (DEST / "results").mkdir(exist_ok=True)
    for name in ["hotpotqa_lora.py", "raw_lora.py", "complete_experiment.py", "progress_dashboard.py", "package_experiment.py", "README.md", "INTERVIEW_GUIDE.md"]:
        shutil.copy2(PROJECT / "training" / name, DEST / "training" / name)
    for name in ["manifest.json", "metrics_baseline.json", "metrics_after.json", "report.json", "training_config.json", "dataset_split_audit.json", "benchmark_selection.json", "paired_analysis.json", "uncertainty.json", "training_data_summary.json", "runtime_versions.json", "adapter_sha256.json"]:
        obj = read(name)
        if "prediction_file" in obj:
            obj["prediction_file"] = "predictions_" + ("after" if "after" in name else "baseline") + ".jsonl"
        if name == "training_config.json":
            obj["data"] = "artifacts/model_training/minicpm5_1b_hotpotqa/mlx_data"
            obj["adapter_path"] = "artifacts/model_training/minicpm5_1b_hotpotqa/adapters"
        (public_run / name).write_text(json.dumps(obj, indent=2) + "\n")
    for name in ["predictions_baseline.jsonl", "predictions_after.jsonl"]:
        shutil.copy2(RUN / name, public_run / name)
    log = re.sub(r"\x1b\[[0-9;]*m", "", (RUN / "training.log").read_text())
    log = log.replace(str(PROJECT), ".")
    (public_run / "training.log").write_text(log)
    (public_run / "workflow_state.json").write_text(json.dumps({"phase": "complete", "report": "experiment_report.md"}, indent=2))
    shutil.copytree(RUN / "figures", DEST / "figures", dirs_exist_ok=True)
    shutil.copytree(RUN / "figures", public_run / "figures", dirs_exist_ok=True)
    (DEST / "REPORT.md").write_text(report)
    (public_run / "experiment_report.md").write_text(report)
    (DEST / "results/examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n")
    snapshot = ui.status()
    if "prediction_file" in snapshot.get("after", {}):
        snapshot["after"]["prediction_file"] = "predictions_after.jsonl"
    (DEST / "results/status.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    (DEST / "results/training-data.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with (DEST / "results/paired_scores.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "type", "before_exact_match", "after_exact_match", "before_token_f1", "after_token_f1", "outcome"])
        for r in examples:
            writer.writerow([r["id"], r["type"], int(r["before"]["exact_match"]), int(r["after"]["exact_match"]), r["before"]["token_f1"], r["after"]["token_f1"], r["outcome"]])
    html = ui.HTML.replace("fetch('/status'", "fetch('./results/status.json'").replace("fetch('/examples'", "fetch('./results/examples.json'")
    html = html.replace("fetch('/training-data'", "fetch('./results/training-data.json'")
    html = html.replace('href="/learn"', 'href="./lesson.html"').replace('href="/guide"', f'href="{REPO}/blob/main/training/INTERVIEW_GUIDE.md"').replace('href="/report"', f'href="{REPO}/blob/main/REPORT.md"')
    html = html.replace("refresh();setInterval(refresh,3000);", "refresh();")
    html = html.replace("Connected · checked ", "Recorded experiment · viewed ")
    html = html.replace("MODEL TRAINING · RESULTS", "RECORDED EXPERIMENT · NOVEMBER 2026")
    html = html.replace('</main>', '<p><a href="'+REPO+'">Source code on GitHub</a> · <a href="'+REPO+'/blob/main/DATA_NOTICE.md">Data attribution and licenses</a></p></main>')
    (DEST / "index.html").write_text(html)
    shutil.copy2(RUN / "fine-tuning-lesson-preview.html", DEST / "lesson.html")
    (DEST / ".nojekyll").write_text("")
    (DEST / ".gitignore").write_text("__pycache__/\n.venv/\n.DS_Store\n*.safetensors\nmlx_data/\nbenchmark_validation*.jsonl\neffective_training_data.jsonl\n")
    (DEST / "requirements.txt").write_text("# Python 3.12 on Apple Silicon for model training\nmlx==0.32.4\nmlx-lm[train]==0.32.2\ndatasets==5.1.0\ntransformers==5.22.1\nhuggingface-hub==2.4.0\nnumpy==2.5.4\n# Only needed to regenerate publication figures\nmatplotlib>=3.9,<4\n")
    (DEST / "DATA_NOTICE.md").write_text("""# Data and model attribution

HotpotQA was created by Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning.

- Dataset: https://huggingface.co/datasets/hotpotqa/hotpot_qa
- Project and paper: https://hotpotqa.github.io/ and https://arxiv.org/abs/1809.09600
- Dataset license: Creative Commons Attribution-ShareAlike 4.0 International, https://creativecommons.org/licenses/by-sa/4.0/

`results/examples.json` contains 500 selected questions, gold answers, supporting-evidence excerpts and generated model responses. The paired prediction files also contain dataset-derived references and generated responses. This comparison collection and dataset-derived excerpts are provided under CC BY-SA 4.0. Changes include sampling, reformatting, evidence excerpts and adding model outputs/scores. Generated responses may reproduce evidence and may be incorrect. Preserve source attribution and indicate further changes if redistributing adapted data.

The code's MIT license does not relicense third-party dataset material. The base model is `openbmb/MiniCPM5-1B`, https://huggingface.co/openbmb/MiniCPM5-1B, Apache-2.0. No model weights or adapters are distributed here. The explanatory lesson is a snapshot made during training; final measurements are in REPORT.md and the results page. Primary method reference: https://arxiv.org/abs/2106.09685 (LoRA).
""")
    (DEST / "LICENSE").write_text("""MIT License

Copyright (c) 2026 Manthan Gopal Dhole

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Third-party and dataset-derived materials are excluded from this license;
see DATA_NOTICE.md.
""")
    (DEST / "README.md").write_text(f"""# MiniCPM5-1B: LoRA fine-tuning on 5,000 HotpotQA examples

A completed local supervised fine-tuning experiment on an Apple M5 Mac with 24 GB RAM, using MLX-LM. It adapts a pretrained 1.08B model to evidence-context QA with LoRA; it does not pretrain a new model from scratch.

**[Interactive results and all 500 before/after comparisons](https://manthan137.github.io/minicpm5-1b-hotpotqa-lora/)** · **[Full report](REPORT.md)** · **[Interview guide](training/INTERVIEW_GUIDE.md)**

| Answer metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Exact match | 15.20% (76/500) | 32.40% (162/500) | +17.20 percentage points |
| Token F1 | 26.925 / 100 | 48.567 / 100 | +21.642 points |

![Before and after on the same questions](figures/before-after.png)

There were **120 newly correct answers and 34 regressions**. The gain is task-specific; 338 answers still fail exact match, and one run does not establish a performance ceiling. See the report for limitations, paired uncertainty, empty-answer failures and proposed next experiments.

## Experiment

- Model: `openbmb/MiniCPM5-1B`; data: `hotpotqa/hotpot_qa`, `distractor`.
- Exactly 5,000 training examples, sampled with seed 137.
- Same 500 saved validation question IDs before and after; sampled from 4,385 already completed baseline questions to shorten evaluation.
- Verified zero overlap in question IDs and normalized question text between training and evaluation.
- Rank 8, scale 20, 16 adapted layers, Adam, learning rate 1e-5, batch size 2, 2,500 updates, one epoch; gradient checkpointing.
- Prompt loss masking; annotated-evidence and gold-answer targets shortened to fit the output budget without changing gold answers.
- Greedy evaluation, 96 new tokens, batch size 8, official HotpotQA answer EM/F1 rules.
- No hyperparameter sweep, confidence calibration or online teacher model.

## Reproduce on Apple Silicon

Use Python 3.12. The recorded package versions are pinned in `requirements.txt`.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python training/hotpotqa_lora.py prepare
python training/hotpotqa_lora.py baseline --overwrite
python training/hotpotqa_lora.py train
python training/hotpotqa_lora.py after --overwrite
python training/hotpotqa_lora.py report
```

The saved `benchmark_selection.json` is reused by preparation. Raw data and model weights download from Hugging Face. The included prediction files define the reported run; reruns can differ with runtime, hardware, batching and upstream model revisions. Inspect `manifest.json` and saved configuration. The training package does not include the base weights, training text or adapter checkpoints.

For the recorded static results page, serve the repository root:

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8000/`. The static page reads saved results; it does not run the model. The separate live dashboard needs freshly prepared benchmark data and runs with `python training/progress_dashboard.py` on port 8767.

## What is included?

- `training/`: preparation, LoRA training, evaluation, dashboard and learning material.
- `artifacts/model_training/minicpm5_1b_hotpotqa/`: manifests, fixed IDs, configuration, split audit, metrics, raw prediction files and cleaned training log.
- `results/`: all paired comparisons, a CSV score table and a static result snapshot.
- `REPORT.md`: method, findings, limitations and headroom discussion.
- `figures/`: publication figures, PNG and SVG.

The public data files reveal this benchmark's answers. Treat it as an explored benchmark for future work and reserve fresh development/final splits before further optimization.

Code: MIT. Dataset-derived comparison material: CC BY-SA 4.0. Model source: Apache-2.0. See [data attribution](DATA_NOTICE.md). Developed with AI assistance; the report documents the implemented method and measured results without claiming unsupported capabilities.
""")
    print(json.dumps({"package": str(DEST), "files": sum(1 for p in DEST.rglob('*') if p.is_file()), "uncertainty": uncertainty}, indent=2))


if __name__ == "__main__":
    main()
