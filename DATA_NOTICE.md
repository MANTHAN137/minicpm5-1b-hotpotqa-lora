# Data and model attribution

HotpotQA was created by Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning.

- Dataset: https://huggingface.co/datasets/hotpotqa/hotpot_qa
- Project and paper: https://hotpotqa.github.io/ and https://arxiv.org/abs/1809.09600
- Dataset license: Creative Commons Attribution-ShareAlike 4.0 International, https://creativecommons.org/licenses/by-sa/4.0/

`results/examples.json` contains 500 selected questions, gold answers, supporting-evidence excerpts and generated model responses. The paired prediction files also contain dataset-derived references and generated responses. This comparison collection and dataset-derived excerpts are provided under CC BY-SA 4.0. Changes include sampling, reformatting, evidence excerpts and adding model outputs/scores. Generated responses may reproduce evidence and may be incorrect. Preserve source attribution and indicate further changes if redistributing adapted data.

The code's MIT license does not relicense third-party dataset material. The base model is `openbmb/MiniCPM5-1B`, https://huggingface.co/openbmb/MiniCPM5-1B, Apache-2.0. No model weights or adapters are distributed here. The explanatory lesson is a snapshot made during training; final measurements are in REPORT.md and the results page. Primary method reference: https://arxiv.org/abs/2106.09685 (LoRA).
