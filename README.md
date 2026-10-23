# BIS-CLIP

**Seeing Expressions, Not Distractions: Background- and Identity-Suppressed CLIP for Facial Expression Recognition**

This repository contains the RAF-DB seven-class configuration associated with
the recorded **92.65%** result (seed 42, epoch 14, zero-based).
It includes Expression-Anchor Dynamic Masking and Identity-Decoupled Optimal
Transport Calibration. This release is limited to the RAF-DB first-stage experiment.

## Installation

Use Python 3.9 and an NVIDIA GPU with a compatible driver. The original runtime
uses PyTorch 1.13.1 and torchvision 0.14.1 with CUDA 11.7.

```bash
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

Run commands from the repository root. Use the bundled `clip` package: it contains
the attention-mask support required by BIS-CLIP.

## Data and Pretrained Model

Obtain [RAF-DB](http://www.whdeng.cn/RAF/model1.html) separately and arrange the
seven-class subset as follows. Dataset images and annotations are not distributed
in this repository.

```text
data/RAF-DB/basic/
  EmoLabel/
    images.txt
    image_class_labels.txt
    train_test_split.txt
  Image/
    aligned_224/
      train_00001.jpg
      ...
```

Each annotation file has two whitespace-separated columns: image ID and value.
`images.txt` maps IDs to filenames, `image_class_labels.txt` maps IDs to labels
1 through 7, and `train_test_split.txt` uses 1 for train and 0 for test. Label
order: surprise, fear, disgust, happiness, sadness, anger, neutral.
Use the same aligned images as the original experiment; the inherited data
preparation follows [CEPrompt](https://github.com/HaoliangZhou/CEPrompt) and
[APViT](https://github.com/youqingxiaozhua/APViT).

Place the pretrained OpenAI CLIP ViT-B/16 model at `checkpoints/ViT-B-16.pt`.
The pretrained model is available from [OpenAI CLIP](https://github.com/openai/CLIP).
Alternatively, pass `--clip-path "ViT-B/16"` to download it through the bundled loader.
Pretrained and trained weights are excluded from Git.

## Training

All training defaults match the archived RAF-DB configuration, except portable
input/output locations. The original parameter record is in
[`configs/rafdb_9265_original.txt`](configs/rafdb_9265_original.txt).

```bash
python train.py --data-path "data/RAF-DB/basic" --clip-path "checkpoints/ViT-B-16.pt"
```

Paths may be absolute. Use `--gpu 0` to choose a GPU and `--record_path outputs`
to choose the output root. Training creates a timestamped run directory.

| Parameter | Value |
| --- | --- |
| Backbone | CLIP ViT-B/16 |
| Seed | 42 |
| Epochs / warmup | 150 / 5 |
| Train / test batch size | 64 / 50 |
| Learning rate / minimum | 5e-6 / 1e-8 |
| Weight decay | 0.0005 |
| Layer decay / frozen layer setting | 0.65 / 2 |
| Distillation weight (`lamda`) | 1.5 |
| OT probability / strength | 0.7 / 0.7 |
| Restitution strength | 0 |
| Identity decorrelation weight | 0.01 |
| Dynamic mask range | 0.05 to 0.25 |
| Anchor protection ratio / graph neighbors | 0.3 / 6 |

This snapshot retains the archived hard attention-mask implementation. Later
soft-mask, feature-similarity, classification-head and AffectNet variants are
not included. Results can vary with hardware, software and data preprocessing.

## Evaluation

```bash
python evaluate.py --data-path "data/RAF-DB/basic" --clip-path "checkpoints/ViT-B-16.pt" --checkpoint "checkpoints/model_epoch_14_acc_0.9265.pth"
```

Evaluation writes accuracy, per-class recall and the confusion matrix under the
output directory. A trained checkpoint must be supplied separately.

## Layout

```text
train.py                  Training entry point and archived defaults
evaluate.py               Checkpoint evaluation entry point
engine.py                 Training and evaluation loops
models/bis_clip.py        BIS-CLIP visual encoder
models/expression_anchor_masking.py  Expression-anchor masking and identity-decoupled OT calibration
clip/                     Modified CLIP implementation and tokenizer vocabulary
dataloader/               RAF-DB loading and original preprocessing
utils/                    Optimizer, scheduler and training helpers
configs/                  Original RAF-DB parameter record
```

## Acknowledgments

The project builds on [CEPrompt](https://github.com/HaoliangZhou/CEPrompt) and
[OpenAI CLIP](https://github.com/openai/CLIP). See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution.
