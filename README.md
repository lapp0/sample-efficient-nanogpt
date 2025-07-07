# Sample-Efficient NanoGPT

## Overview
Sample-Efficient NanoGPT is a fork of [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt). The goal is to train a language model that achieves ≤ 3.28 cross-entropy loss on the FineWeb validation set. No specific hardware requirements.

**Why sample efficiency?**
Maximizing learning per token uncovers new architectures and training methods that current hardware and kernels don’t yet optimize. These insights steer future hardware and kernel design to leverage these advancements.

## Running the Current Record
**Setup**
```bash
git clone https://github.com/lapp0/sample-efficient-nanogpt.git
cd sample-efficient-nanogpt
pip install -r requirements.txt
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
python data/cached_fineweb10B.py 10
```

**Run**
```bash
export N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
torchrun --standalone --nproc_per_node=$N_GPU train_gpt.py
```

## World Record History

| Implementation          | Tokens    | Date       | Note                                                                |
|-------------------------|-----------|------------|---------------------------------------------------------------------|
| llm.c (baseline)        | 10,486 M  | 05/28/2024 | Uses tied embeddings, only 124M parameters                          |
| ~~modded-nanogpt~~      | ~~696 M~~ | 05/25/2025 | Invalid, uses value embeddings, increasing total parameters to 201M |
| No Value Embedding      | 747.11M   | 07/06/2025 | Remove value embeddings, train for 1900 steps                       |
| Recurrent Intermediates | 668.47M   | 07/07/2025 | Repeat layers in forward pass                                       |

## Rules

* **Parameter limit**: ≤ 162M parameters (including embeddings). Inactive parameters count towards the total.
* **Target**: achieve ≤ 3.28 cross-entropy loss on FineWeb val.
* **Data**: Must use FineWeb dataset. Sample order are fixed. Samples cannot be repeated. Sample size per batch may vary.
* **Time**: Must train on 8xH100 in fewer than 30 minutes

## Records

### No Value Embedding
Remove value embeddings from modded-nanogpt in order to comply with 162M parameter limit. Requires longer training run of 1900 steps

### Recurrent Intermediate Layers
[logs](logs/83dda42f-c076-49b7-8933-457f56f0f4b0.txt)

Rather than the forward pass looping through layers [0, 1, ..., 11], instead layers are looped. Specifically, the pattern is
```
0, 1, 2,
3, 4, 5, 6, 7, 8,
3, 4, 5, 6, 7, 8,
3, 4, 5, 6, 7, 8,
3, 4, 5, 6, 7, 8,
9, 10, 11
```

`step:1700/1700 val_loss:3.2701 train_time:6555576ms step_avg:3856.22ms tokens:668.47M`

## Citations

\[1]: [Keller Jordan et al. *modded-nanogpt: Speedrunning the NanoGPT baseline*.](https://github.com/KellerJordan/modded-nanogpt/)
