# Image Captioning From Scratch

> Automatically generate natural language descriptions from images — without any pretrained weights.

<p align="center">
  <code>Image 224×224</code> → <b>CNN Encoder</b> (6 blocks) → <code>512-d vector</code> → <b>LSTM Decoder</b> (2 layers) → <code>"a dog running in the park"</code>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Dataset-MS%20COCO%202017-blue" alt="COCO">
  <img src="https://img.shields.io/badge/GPU-Kaggle%20T4%2FP100-76B900?logo=nvidia&logoColor=white" alt="GPU">
  <img src="https://img.shields.io/badge/From%20Scratch-100%25-orange" alt="From Scratch">
</p>

---

## Table of Contents

- [Objective](#objective)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Key Techniques](#key-techniques)
- [Hyperparameters](#hyperparameters)
- [Results](#results)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)

---

## Objective

Design and train an **encoder-decoder** model capable of generating an English sentence describing the content of an image.

**Main constraint**: the entire model is built **from scratch** — no transfer learning, no pretrained weights (no ResNet, no ImageNet features). Both the CNN and the LSTM are trained from zero.

---

## Architecture

```
┌─────────────┐     ┌───────────────────┐     ┌──────────┐     ┌──────────────────┐     ┌──────────┐
│   Image     │────▶│   CNN Encoder     │────▶│  Feature  │────▶│   LSTM Decoder   │────▶│ Caption  │
│  224×224×3  │     │  6 VGG-like blocks│     │  512-d    │     │   2 layers       │     │  text    │
└─────────────┘     └───────────────────┘     └──────────┘     └──────────────────┘     └──────────┘
```

### CNN Encoder

A **VGG-style** CNN with **6 convolutional blocks** and **12 Conv2d layers**:

| Block | Channels | Resolution |
|-------|----------|------------|
| Block 1 | 3 → 32 | 224 → 112 |
| Block 2 | 32 → 64 | 112 → 56 |
| Block 3 | 64 → 128 | 56 → 28 |
| Block 4 | 128 → 256 | 28 → 14 |
| Block 5 | 256 → 512 | 14 → 7 |
| Block 6 | 512 → 512 | 7 → 3 |

Each block: `Conv2d(3×3)` → `BatchNorm` → `ReLU` → `Conv2d(3×3)` → `BatchNorm` → `ReLU` → `MaxPool2d(2×2)`

Followed by: `AdaptiveAvgPool2d(1,1)` → `Dropout(0.3)` → `Linear(512, 512)` → `BatchNorm1d`

### LSTM Decoder

- **Embedding**: vocab_size → 256 dimensions (padding_idx=0)
- **LSTM**: 2 stacked layers, hidden_dim=512, dropout=0.35
- **Projection**: Linear(512 → vocab_size) → logits
- **Initialization**: the image feature vector initializes h₀ and c₀ via two linear projections

---

## Dataset

**MS COCO 2017** — the standard benchmark for image captioning.

| | Value |
|---|---|
| Training images | ~118,000 |
| Validation images | ~5,000 |
| Captions used / image | 2 (out of 5 available) |
| Captions for evaluation | 5 (all references) |
| Vocabulary size | ~9,000 words |
| Minimum frequency threshold | 5 occurrences |

The validation set is split 50/50 **by image** (not by caption) to create the validation and test sets.

---

## Key Techniques

### Pack Padded Sequences
The LSTM ignores `<pad>` tokens via `pack_padded_sequence`, preventing the model from learning to predict padding and improving gradient quality.

### Scheduled Teacher Forcing
The teacher forcing ratio decreases linearly from **1.0 → 0.6** over 30 epochs. The model progressively learns to handle its own prediction errors, reducing exposure bias.

### Label Smoothing (ε = 0.1)
Targets are no longer one-hot but smoothed: 0.9 for the correct word, 0.1 distributed over the rest of the vocabulary. Prevents overconfidence and improves generalization.

### Data Augmentation
Full augmentation pipeline at training time (essential since the CNN is trained from scratch):
- `Resize(240)` → `RandomCrop(224)` → `RandomHorizontalFlip(0.5)` → `RandomRotation(±10°)` → `ColorJitter(±20%)` → `Normalize`

### Beam Search
Inference with beam search (k=3 or k=5) and length-normalized scoring to avoid favoring short captions.

### Multi-Reference BLEU Evaluation
BLEU-4 score implemented from scratch, evaluated by taking the best score across all 5 reference captions for each image.

---

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `IMG_SIZE` | 224 | Richer visual features |
| `BATCH_SIZE` | 32 | Trade-off between GPU memory and gradient stability |
| `LEARNING_RATE` | 3e-4 | "Karpathy constant" — excellent default for Adam |
| `NUM_LSTM_LAYERS` | 2 | More capacity for COCO's vocabulary |
| `EMBED_DIM` | 256 | Word embedding dimension |
| `HIDDEN_DIM` | 512 | LSTM hidden state dimension |
| `DROPOUT` | 0.3 / 0.35 | Encoder / Decoder |
| `LABEL_SMOOTHING` | 0.1 | Loss regularization |
| `TF_RATIO` | 1.0 → 0.6 | Decreasing teacher forcing |
| `GRAD_CLIP` | 5.0 | Prevents LSTM gradient explosion |
| `NUM_EPOCHS` | 30 | With ReduceLROnPlateau (factor=0.5, patience=3) |

### Weight Initialization

| Layer | Method |
|-------|--------|
| Conv2d | Kaiming Normal (fan_out, ReLU) |
| BatchNorm | γ=1, β=0 |
| Embedding | Uniform [-0.1, 0.1] |
| LSTM input-hidden | Xavier Uniform |
| LSTM hidden-hidden | Orthogonal |
| Forget gate bias | 1.0 |
| fc_out | Xavier Uniform |

---

## Results

The model is evaluated on 200 test set images using beam search (k=3).

BLEU-4 metrics are computed in both single-reference and multi-reference mode (max over the 5 captions per image).

Sample generated captions:

```
Image: [photo of a dog in a park]
Greedy  : "a dog is standing in the grass"
Beam(3) : "a dog is running through the grass in a park"
Beam(5) : "a brown dog running through a green field"
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.4.1
- torchvision 0.19.1
- GPU recommended (trained on Kaggle T4/P100)

### On Kaggle (recommended)

1. Create a new notebook on [Kaggle](https://www.kaggle.com/)
2. Add the dataset **`awsaf49/coco-2017-dataset`** via the "Add Data" button
3. Enable GPU (Settings → Accelerator → GPU)
4. Import the notebook and run all cells

### Locally

```bash
# Clone the repo
git clone https://github.com/<your-username>/image-captioning-from-scratch.git
cd image-captioning-from-scratch

# Install dependencies
pip install torch==2.4.1 torchvision==0.19.1

# Download MS COCO 2017
# - Images: https://cocodataset.org/#download
# - Annotations: captions_train2017.json, captions_val2017.json

# Update the paths in the Config class, then run the notebook
```

### Resuming Training

The checkpoint system automatically saves the best model (based on val_loss). To resume an interrupted training session, simply rerun the notebook — the script detects the checkpoint and picks up where it left off.

---

## Project Structure

```
.
├── notebook.ipynb            # Main notebook (training + evaluation)
├── README.md
├── checkpoint_coco/
│   └── best_model.pth        # Best model checkpoint
└── assets/                   # Images for the README (optional)
```

---

## Summary

| Component | Choice | Details |
|-----------|--------|---------|
| **Dataset** | MS COCO 2017 | ~118k images, 2 captions/image for training |
| **Encoder** | CNN 6 blocks (12 conv) | VGG-like, from scratch, BatchNorm + ReLU |
| **Decoder** | LSTM 2 layers | Pack padded, scheduled teacher forcing |
| **Loss** | CrossEntropy | label_smoothing=0.1, ignore_index=0 |
| **Optimizer** | Adam | lr=3e-4 + ReduceLROnPlateau |
| **Inference** | Beam Search | k=3 and k=5, length-normalized scoring |
| **Metric** | BLEU-4 | Multi-reference, implemented from scratch |
| **Platform** | Kaggle GPU | T4/P100, checkpoint resume |

---

<p align="center">
  <i>2nd Year Engineering Project — Deep Learning</i>
</p>
