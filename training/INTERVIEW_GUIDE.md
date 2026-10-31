# Fine-tuning a pretrained model: study and interview guide

This guide describes the experiment actually implemented in this project. Training and both evaluations are complete. The same 500-question benchmark improved from 15.2% to 32.4% exact match and from 26.925 to 48.567 token F1. Consult the final experiment report for the paired results and limitations.

## Your 45-second explanation

> I set up a local supervised fine-tuning experiment for context-based question answering using the pretrained MiniCPM5-1B model. I used LoRA to train a small adapter instead of updating the base weights. I selected 5,000 HotpotQA training examples and kept 500 validation questions separate, checking that neither question IDs nor normalized question texts overlapped. The teaching targets combined annotated supporting evidence with the correct answer. I trained rank-8 adapters on 16 layers for one epoch, using batch size two and a learning rate of 1e-5. I compared answer exact match and token F1 on the same 500 questions before and after training.

**Measured result sentence:** The original model scored 15.2% exact match and 26.925 token F1. After training, exact match was 32.4% and F1 was 48.567, gains of 17.2 and 21.642 percentage points respectively on the same 500 questions. There were 120 newly correct answers and 34 regressions.

Describe your own role accurately. You can explain the experiment you ran, the decisions you understand, and the implementation developed with AI assistance. Do not imply that you independently invented LoRA, pretrained the base model, or personally wrote every line if that is not what happened.

## 1. What are we training?

The starting model is `openbmb/MiniCPM5-1B`, with 1,080,632,832 parameters. A parameter is a learned number used in the model's computations. Its original training already gave it language abilities. This project adapts that pretrained model to a narrower task.

**Pretraining** generally builds broad capabilities from large corpora. **Fine-tuning** starts from existing weights and adapts behavior using additional examples. **Inference** uses the model to generate an answer without changing weights. **Evaluation** measures inference results against references.

Our method is **supervised fine-tuning (SFT) with low-rank adaptation (LoRA)**. It is not reinforcement learning, training from scratch, or QLoRA. We have not applied a quantized training configuration.

## 2. What task did we choose?

HotpotQA's `distractor` configuration supplies a question and Wikipedia passages, including distracting passages. Answering often requires connecting evidence from multiple passages.

The input is a system instruction, question, and context passages. The output is a brief evidence-based response ending with `Answer: <short answer>`. We evaluate the answer text. We do not compute HotpotQA's supporting-fact or joint metrics.

This is **context-based QA**. A result here does not establish better closed-book factual recall, broad reasoning, conversation, or every kind of general knowledge. We are not building a retrieval system in this experiment: the dataset supplies the context.

## 3. How is the data separated?

| Purpose | Count | Source | Used for gradient updates? |
| --- | ---: | --- | --- |
| Training | 5,000 | Official training split | Yes |
| Before/after benchmark | 500 | Official validation split | No |

Training examples were sampled with seed 137. Initially, a full 7,405-question baseline was started. At the user's request, we stopped that run and sampled 500 questions, with seed 137, from the 4,385 validation questions already completed. Selection did not use correctness to choose easier or harder examples. Saved IDs fix the exam for both evaluations.

We verified **zero overlapping IDs and zero overlapping normalized question texts** between the actual 5,000 training prompts and the 500 benchmark questions. The check also reconstructed the selected training IDs and matched their hash to the manifest.

This checks leakage in our fine-tuning data. It does not establish that the base model never encountered public HotpotQA material in its original pretraining. Different questions may also share Wikipedia facts or passages.

The dataset calls this split “validation.” In our current experiment, the selected 500 questions function as the held-out comparison benchmark. If we later tune hyperparameters repeatedly, we should use a separate development set and preserve a final test set for the final measurement. Repeatedly selecting settings on these 500 answers would make the reported comparison more optimistic.

## 4. What does the model learn from?

The dataset provides annotated supporting sentences and a gold answer. We form a supervised response from that evidence and the answer. Long evidence is shortened to an excerpt so the entire target fits below the 96-token output budget; the gold answer stays intact.

The prompt is rendered and encoded once. We found that MLX's ordinary prompt/completion loader would add another chat template around the already rendered prompt, so a custom loader avoids that duplication.

**Prompt masking** means prompt positions do not contribute directly to the loss. They still provide context for predicting the target tokens. Loss is computed on the supervised response, including its end-of-sequence token.

**Teacher forcing** means the model receives the correct preceding target tokens when predicting each next target token during training. It does not mean a second teacher model is generating explanations. At inference time, the model receives its own generated tokens instead.

We are not generating a complete wrong answer, diagnosing it with another model, and retraining on that diagnosis. This implementation uses supervised evidence and answer targets already present in the dataset. It does not expose or train on a teacher's private chain of thought.

## 5. How does one update work?

1. Load two examples and their tokenized prompts and targets.
2. Run a forward pass to obtain next-token probabilities.
3. Compute cross-entropy loss on target positions. For one correct token, the loss is `-log(p(correct token))`; the implementation averages across contributing tokens.
4. Backpropagate to compute gradients for the trainable adapter factors.
5. Use Adam and learning rate 1e-5 to update those factors.
6. Repeat until 2,500 updates have processed the 5,000 examples once.

The original base weights remain frozen. Gradients must still flow through relevant model computations, so freezing weights does not remove all forward and backward compute.

## 6. Why LoRA?

For a conceptual weight matrix `W`, LoRA adds a low-rank update: `W_effective = W_frozen + scaled(B × A)`. The dimensions and multiplication order depend on matrix conventions. Two thin factors replace a dense trainable update. Rank controls their inner dimension and therefore the adapter's capacity and size.

In our MLX implementation, the forward operation adds `scale × ((x @ A) @ B)` to the base linear output. The saved scale is 20; do not automatically call that value “alpha” or assume another library's alpha/rank convention.

This run trains about **3.736 million parameters**, approximately **0.346% of the base-model parameter count**, on selected transformations in 16 layers. The downloaded base model remains necessary at inference together with the adapter, unless the adapter is merged into the base weights later. We have not performed that merge or an Android export here.

We chose LoRA to keep local task adaptation and optimizer state manageable. It reduces trainable memory compared with full fine-tuning. It does not guarantee the same quality as full fine-tuning for every task, eliminate all memory use, or guarantee that the base model avoids behavioral regressions.

## 7. Exact settings and their purpose

| Setting | Actual value | Meaning and reason |
| --- | --- | --- |
| Training examples | 5,000 | Fixed experiment size |
| Epochs | 1 | One initial pass; limits repeat exposure |
| Batch size | 2 | Two examples per update; conservative local memory use |
| Updates | 2,500 | `5,000 / 2`; not 2,500 epochs |
| Optimizer | Adam | Uses gradient estimates to update adapter factors |
| Learning rate | 0.00001 | Conservative update size; not empirically established as optimal |
| LoRA rank | 8 | Compact trainable factors |
| LoRA scale | 20 | MLX multiplier for the adapter contribution |
| LoRA dropout | 0 | No adapter dropout in this run |
| Adapted layers | 16 | Subset of model layers receive trainable LoRA factors |
| Sequence limit | 4,096 tokens | Fits complete prepared examples; code refuses to truncate an overlong example |
| Gradient checkpointing | Enabled | Recomputes some activations during backpropagation to save memory, with extra compute |
| Random seed | 137 | Records reproducible sampling and initialization choices |
| Checkpoint interval | 100 updates | Saves intermediate adapter weights |

A **token** is a tokenizer unit, often a word fragment, punctuation, or part of a word. A 4,096-token limit is not a 4,096-word limit. A **checkpoint** is saved weights; gradient **checkpointing** is a separate memory technique.

These were selected as a starting configuration. We have not run a hyperparameter sweep, an ablation study, a comparison against full fine-tuning, or confidence calibration. Do not call the settings optimal or “fully calibrated.”

**Probability calibration** is different: for predictions assigned 80% confidence, roughly 80% should be correct over an appropriate set. We have not measured reliability diagrams, expected calibration error, or fitted temperature scaling. Evaluation temperature zero here means greedy decoding; it is not confidence calibration.

## 8. How do the metrics work?

**Answer exact match (EM):** lowercase and remove punctuation, articles, and extra spaces, then compare prediction with reference. A score of 15.2% means 76 of the 500 answers matched.

**Answer token F1:** split normalized answers into words, compute word-overlap precision and recall, and combine them using `2PR/(P+R)`. This is an answer-word metric, not overlap between the model's internal subword token IDs. The per-question F1 scores are averaged.

Example: gold `New York City`, prediction `New York`. Precision is 2/2; recall is 2/3. F1 is 0.8, or 80/100. EM is zero. For yes/no/noanswer cases, HotpotQA's special rules require the normalized answer to match rather than award partial overlap.

Baseline: **15.2% EM and 26.9249769 token F1**. The latter does not mean that 26.9% of questions were fully correct. Answer extraction uses the final `Answer:` line, with a fallback if that marker is missing; parsing failures are recorded for inspection. Format compliance can therefore affect scores.

## 9. What makes the comparison fair?

Both evaluations use the same 500 question IDs, supplied evidence, prompt, answer extraction, normalization, output limit, and scoring. Evaluation uses greedy decoding, a maximum of 96 new tokens, and batch size eight. Reference answers are for scoring, not appended as evaluation answers in the input.

The after-training run loads the final adapter on the same base checkpoint. It uses the planned final epoch checkpoint rather than selecting a checkpoint by whichever gives the best benchmark result.

Report absolute scores and **percentage-point changes**. A hypothetical move from 15.2% to 20.2% is +5 percentage points and about 32.9% relative improvement. Those are different descriptions. This is a math example, not our observed result.

Five hundred questions give a limited estimate. Each additional fully correct answer changes EM by 0.2 percentage points. Questions are evaluated in pairs before and after, so a paired bootstrap or a suitable paired test is preferable to treating the two measurements as unrelated. We have not yet implemented an uncertainty analysis or repeated training seeds.

## 10. Does falling training loss mean success?

It means the adapter is fitting the supervised training targets. It does not guarantee higher held-out accuracy. A model can overfit, learn response formatting without improving factual understanding, or regress on other tasks.

If final EM or F1 falls, report the regression. Examine paired errors, missing final answers, truncation, evidence quality, and category breakdowns. For another experiment, use a separate development set to compare settings. Do not quietly keep changing the model against the same final exam until a desirable score appears.

## 11. Questions to practice aloud

**Why not train from scratch?** We wanted to adapt existing language capabilities with a small dataset and local compute. Pretraining a competitive language model is a much larger project.

**Why LoRA instead of full fine-tuning?** It reduces trainable parameters and optimizer memory, making local adaptation practical. We did not empirically establish that it is the best method for this task.

**What did you call a teacher?** The gold answers and annotated supporting evidence provide supervision. Teacher forcing supplies the correct previous target tokens. There is no separate online teacher model in this pipeline.

**How did you prevent leakage?** Separate official splits, fixed saved benchmark IDs, reconstructed training-ID hash, and zero-overlap checks on IDs and normalized question text. Public pretraining contamination remains unknown.

**Why mask the prompt?** We want the supervised loss to reward the target response, while allowing the input question and passages to provide context.

**What did you calibrate?** We chose a conservative training configuration and standardized evaluation. We did not perform confidence calibration or an empirical hyperparameter search.

**Why only 500 evaluation examples?** To shorten generation time while preserving a paired before/after comparison. It is a sample estimate, not the official full-validation leaderboard score.

**What is the main limitation?** This experiment evaluates evidence-context QA on one public dataset, one training seed, and one sample. It cannot prove general intelligence, broad closed-book knowledge, or a production-ready Android model.

**How much did it improve?** Give only the measured final report numbers. Until they exist, say the post-training result is pending.

## Artifacts to show in an interview

- `training/hotpotqa_lora.py`: preparation, evaluation, training launch, and report generation.
- `training/raw_lora.py`: single-pass prompt encoding, loss masking, shortened teaching targets, and LoRA training.
- `artifacts/model_training/minicpm5_1b_hotpotqa/manifest.json`: model/data identifiers, seed and training-ID hash.
- `benchmark_selection.json`: fixed 500 benchmark IDs.
- `dataset_split_audit.json`: actual zero-overlap verification.
- `effective_training_data.jsonl`: exact training prompts and targets.
- `training_config.json`, `training.log`, and `adapters/`: settings, progress, and saved weights.
- `predictions_baseline.jsonl` and eventually `predictions_after.jsonl`: per-question evidence for the comparison.
- `metrics_baseline.json`, eventually `metrics_after.json`, `report.json`, and `experiment_report.md`: final measurements and documentation.

All artifact paths above are relative to the experiment directory unless prefixed with `training/`.

## Primary sources

- [MiniCPM5-1B model](https://huggingface.co/openbmb/MiniCPM5-1B)
- [HotpotQA dataset](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
- [Original LoRA paper](https://arxiv.org/abs/2106.09685)
- [MLX-LM LoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- [Official HotpotQA scoring implementation](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)
