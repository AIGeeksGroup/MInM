# MInM: Mask Instance Modeling for Visual Representation Learning

This is the official repository for the paper:

> **MInM: Mask Instance Modeling for Visual Representation Learning**
>
> Yihao Wan\*, [Author 2]\*, [Author 3]†
>
> \*Equal contribution. †Corresponding author.
>
> ### [Paper](https://arxiv.org/) | [HF Paper](https://huggingface.co/papers/)

> [!NOTE]
> _⚠️ **Repository Structure**: This repo contains the **MInM pre-training framework** (extending MAE with instance-guided masking) and evaluation pipelines for **ImageNet**, **ImageNette**, and **ImageWoof**._

## Citation

If you find our work useful, please cite:

```bibtex
@article{wan2026minm,
  title={MInM: Mask Instance Modeling for Visual Representation Learning},
  author={Wan, Yihao and others},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
```


## Introduction

Standard Masked Autoencoders (MAE) rely on **random masking**, which treats all patches equally regardless of semantic content. This leads to suboptimal representation learning — the model spends equal effort reconstructing uninformative background regions and semantically rich object regions.

We present **MInM (Mask Instance Modeling)**, an instance-guided masking strategy that leverages SAM-generated instance segmentation masks to provide **semantic and structural guidance** during masked image modeling. Key contributions include:

- **Instance-Guided Masking**: Replacing random masking with patch-level masks derived from SAM instance segmentation, forcing the model to reconstruct semantically coherent object regions.
- **Patch-Level Mask Aggregation**: A differentiable module (`InstanceGuidedMasking`) that converts pixel-level binary masks to patch-level masking decisions via spatial averaging and thresholding.
- **Improved Representation Quality**: Demonstrating gains in linear probing and fine-tuning accuracy on ImageNet, ImageNette, and ImageWoof benchmarks.

## ⚙️ Installation

### 1. Environment Setup

```bash
# 1. Create environment
conda create -n minm python=3.10 -y
conda activate minm

# 2. Install dependencies
pip install -r requirements.txt
```

### 2. User Configuration (📌 Input Required)

You must configure API keys and data paths before running experiments.

**Option A: Environment Variables**

```env
# --- For W&B Experiment Tracking ---
WANDB_API_KEY="your-wandb-key"
```

**Option B: Dataset Preparation**

- **ImageNet-1K**: Download and organize into `train/` and `val/` directories following the [PyTorch ImageNet format](https://pytorch.org/vision/stable/datasets.html#imagenet).
- **ImageNette / ImageWoof**: Automatically downloaded and extracted by the training scripts from HuggingFace.

## 🔧 Instance Mask Generation

Generate SAM-based instance masks for your dataset. This is a prerequisite for MInM pre-training.

> _Requires the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)._

```bash
# Generate instance masks for ImageWoof
python tools/generate_sam_masks.py \
  --dataset imagewoof \
  --output_dir data/imagewoof/instance_masks
```

## 🧪 Experiments & Reproducibility

### 1. MInM Pre-training

Pre-train a ViT with instance-guided masking on ImageNette or ImageWoof.

```bash
# MInM on ImageNette (ViT-Base, 125 epochs)
python tools/train_imagenette.py \
  --epochs 125 \
  --batch_size 32 \
  --blr 1.5e-3 \
  --model mae_vit_base_patch16

# MInM on ImageWoof (ViT-Base, 125 epochs)
python tools/train_imagewoof.py \
  --epochs 125 \
  --batch_size 32 \
  --blr 1.5e-3 \
  --model mae_vit_base_patch16
```

### 2. MAE Baseline Pre-training

Train the standard MAE baseline for comparison.

```bash
# MAE on ImageNette
python tools/train_imagenette_mae.py \
  --epochs 125 \
  --batch_size 32 \
  --blr 1.5e-3 \
  --model mae_vit_base_patch16

# MAE on ImageNet-1K (multi-node)
python tools/submitit_pretrain.py \
  --job_dir ./output \
  --nodes 8 \
  --batch_size 64 \
  --model mae_vit_large_patch16 \
  --mask_ratio 0.75 \
  --norm_pix_loss \
  --epochs 800
```

### 3. Linear Probing

Evaluate the quality of learned representations by training a linear classifier on frozen features.

```bash
python tools/main_linprobe.py \
  --batch_size 512 \
  --model vit_base_patch16 \
  --finetune /path/to/pretrain_checkpoint.pth \
  --epochs 90 \
  --data_path /path/to/imagenet
```

### 4. Fine-tuning

Fine-tune the pre-trained model end-to-end for downstream classification.

```bash
python tools/main_finetune.py \
  --batch_size 32 \
  --model vit_base_patch16 \
  --finetune /path/to/pretrain_checkpoint.pth \
  --epochs 100 \
  --data_path /path/to/imagenet
```

## 📊 Results

### Pre-trained Checkpoints

| | ViT-Base | ViT-Large | ViT-Huge |
|:---|:---:|:---:|:---:|
| MAE (Baseline) | [download](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) | [download](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth) | [download](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth) |
| MInM (Ours) | coming soon | coming soon | coming soon |

### ImageNet-1K Classification

| Method | ViT-B (Linear Probe) | ViT-B (Fine-tune) |
|:---|:---:|:---:|
| MAE | — | 83.6 |
| MInM (Ours) | — | — |

> _Results will be updated upon paper acceptance._

## 📁 Directory Structure

```text
.
├── assets/                  # Images for README
├── models/                  # Model architectures
│   ├── models_mae.py        # Standard MAE (ViT encoder-decoder)
│   ├── models_minm.py       # MInM with InstanceGuidedMasking
│   └── models_vit.py        # Vision Transformer utilities
├── engine/                  # Training & evaluation loops
│   ├── engine_pretrain.py   # MAE pre-training loop
│   ├── engine_pretrain_minm.py  # MInM pre-training loop
│   ├── engine_finetune.py   # Fine-tuning loop
│   └── engine_probing.py    # Linear probing evaluation
├── tools/                   # Entry-point scripts
│   ├── train_imagenette.py  # MInM training on ImageNette
│   ├── train_imagewoof.py   # MInM training on ImageWoof
│   ├── train_imagenette_mae.py  # MAE baseline on ImageNette
│   ├── generate_sam_masks.py    # SAM instance mask generation
│   ├── main_pretrain.py     # MAE pre-training (ImageNet)
│   ├── main_finetune.py     # Fine-tuning script
│   ├── main_linprobe.py     # Linear probing script
│   └── submitit_*.py        # Distributed training wrappers
├── util/                    # Utilities (LR scheduling, LARS, etc.)
├── data/                    # Dataset storage
│   ├── imagenette/          # ImageNette + SAM masks
│   └── imagewoof/           # ImageWoof + SAM masks
├── configs/                 # Configuration YAMLs
├── docs/                    # Additional documentation
│   ├── PRETRAIN.md          # Pre-training instructions
│   └── FINETUNE.md          # Fine-tuning instructions
├── demo/                    # Visualization demos
├── output/                  # Checkpoints & logs
├── requirements.txt         # Python dependencies
└── LICENSE                  # CC-BY-NC 4.0
```

## Acknowledgements

We acknowledge the use of the following resources:

- [**MAE**](https://github.com/facebookresearch/mae): Masked Autoencoders Are Scalable Vision Learners.
- [**SAM**](https://github.com/facebookresearch/segment-anything): Segment Anything Model for instance mask generation.
- [**DeiT**](https://github.com/facebookresearch/deit): Data-efficient Image Transformers.
- [**timm**](https://github.com/rwightman/pytorch-image-models): PyTorch Image Models.

## License

This project is licensed under the [CC-BY-NC 4.0 License](LICENSE).
