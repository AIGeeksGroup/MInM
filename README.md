# MInM: Mask Instance Modeling for Visual Representation Learning

This is the official repository for the paper:

> **MInM: Mask Instance Modeling for Visual Representation Learning**
>
> Yiran Wang<sup>b,1</sup>, Junlin Long<sup>b,1</sup>, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)<sup>a,2</sup>, Rong Fu<sup>c</sup>, Ruicheng Zhang<sup>d</sup>, Rundong Xue<sup>e</sup>, Zirui Song<sup>f</sup>, Renda Han<sup>g</sup>, Hoi Leong Lee<sup>h</sup>, Xiuying Chen<sup>f</sup> and Yang Zhao<sup>a,*</sup>
>
> <sup>a</sup>La Trobe University &nbsp; <sup>b</sup>University of Sydney &nbsp; <sup>c</sup>University of Macau &nbsp; <sup>d</sup>Tsinghua University &nbsp; <sup>e</sup>Xi'an Jiaotong University &nbsp; <sup>f</sup>MBZUAI &nbsp; <sup>g</sup>Tianjin University &nbsp; <sup>h</sup>Universiti Malaysia Perlis
>
> <sup>1</sup>Equal contribution, co-first authors. <sup>2</sup>Project lead. <sup>*</sup>Corresponding author.
>
> ### [Paper](https://arxiv.org/) | [HF Paper](https://huggingface.co/papers/) | [Model](https://huggingface.co/AIGeeksGroup/MInM)

> [!NOTE]
> _⚠️ **Repository Structure**: This repo contains the **MInM pre-training framework** (extending MAE with instance-guided masking) and evaluation pipelines for **ImageNet-1K**, **Pascal VOC**, and **Imagenette**._

## Citation

If you find our work useful, please cite:

```bibtex
@article{wang2026minm,
  title={MInM: Mask Instance Modeling for Visual Representation Learning},
  author={Wang, Yiran and Long, Junlin and Zhang, Zeyu and Fu, Rong and Zhang, Ruicheng and Xue, Rundong and Song, Zirui and Han, Renda and Lee, Hoi Leong and Chen, Xiuying and Zhao, Yang},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
```


## Introduction

Masked image modeling (MIM) has emerged as a powerful self-supervised learning paradigm in computer vision, inspired by the success of masked language modeling in NLP. By masking parts of the input image and training the model to reconstruct the missing content, MIM enables the learning of rich and transferable visual representations without requiring manual annotations. Recent methods such as [MAE](https://arxiv.org/abs/2111.06377), [BEiT](https://arxiv.org/abs/2106.08254), and [SimMIM](https://arxiv.org/abs/2111.09886) have demonstrated strong performance on large-scale benchmarks.

Despite their success, existing MIM methods predominantly rely on **random masking** strategies that treat all image regions equally, regardless of their semantic content. Common prediction targets, such as pixel-level or discrete tokens, often fail to align with human perception, leading to semantically ambiguous representations. As a result, the model may allocate excessive capacity to reconstructing redundant background content, weakening its ability to learn representations useful for downstream tasks.

We present **MInM (Mask Instance Modeling)**, a novel masked image modeling framework that leverages **instance-aware saliency masks** to guide visual representation learning. Instead of applying uniformly distributed random occlusion, MInM deliberately identifies foreground instance areas derived from [SAM2](https://arxiv.org/abs/2408.00714) as the primary reconstruction objective. Built upon the MAE architecture, MInM integrates a task-aligned masking pipeline that improves both global and localized representation quality — **without any modifications to the encoder or decoder**.

Key contributions:

- **Instance-Guided Masking Framework**: We introduce MInM, a novel instance-guided masked image modeling framework that leverages semantic masks to enhance visual representation learning.
- **Task-Aligned Masking Strategy**: We propose a masking strategy based on high-quality instance segmentation masks from SAM2, which encourages the model to reconstruct foreground content while ignoring background redundancy.
- **Extensive Validation**: We validate the effectiveness of MInM across multiple datasets, including ImageNet-1K, Pascal VOC, and Imagenette.

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
- **ImageNette**: Automatically downloaded and extracted by the training scripts from HuggingFace.
- **Pascal VOC**: Download VOC 2007 + VOC 2012 and organize following [MMDetection format](https://mmdetection.readthedocs.io/).

## 🔧 Instance Mask Generation

Generate SAM-based instance masks for your dataset. This is a prerequisite for MInM pre-training.

> _Requires [SAM2](https://github.com/facebookresearch/segment-anything-2)._

```bash
# Generate instance masks for ImageNet-1K
python tools/generate_sam_masks.py \
  --dataset imagenet \
  --output_dir data/imagenet/instance_masks

# Generate instance masks for Imagenette
python tools/generate_sam_masks.py \
  --dataset imagenette \
  --output_dir data/imagenette/instance_masks
```

## 🧪 Experiments & Reproducibility

### 1. MInM Pre-training

Pre-train a ViT with instance-guided masking.

```bash
# MInM on ImageNet-1K (ViT-Base, 600 epochs, multi-node)
python tools/imagenet_1kminm_parallel.py \
  --epochs 600 \
  --batch_size 256 \
  --blr 5e-4 \
  --model mae_vit_base_patch16 \
  --warmup_epochs 80 \
  --data_path /path/to/imagenet

# MInM on Imagenette (ViT-Base, 100 epochs)
python tools/train_imagenette.py \
  --epochs 100 \
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

### ImageNet-1K Linear Probing (ViT-B/16, 600 epochs)

| Method | Top-1 (%) | Top-5 (%) |
|:---|:---:|:---:|
| MAE (Baseline) | 53.15 | 80.59 |
| MInM (Ours, best) | 38.25 | **81.50** |

> MInM surpasses MAE in Top-5 accuracy under the same training protocol, indicating stronger semantic coverage. See the paper for detailed hyperparameter ablations (Table 3).

### Pascal VOC 2007 Object Detection (mAP %)

| Method | mAP |
|:---|:---:|
| Faster R-CNN + R50-FPN (ours) | **75.3** |
| Faster R-CNN + MInM ViT (ours) | 34.5 |
| Faster R-CNN + MAE ViT (ours) | 33.5 |

> MInM-ViT consistently outperforms MAE-ViT across the majority of semantic categories, with particular prominence on categories such as *cat*, *dog*, and *sofa*.

### Imagenette Linear Probing (ViT-B/16)

| Method | Epochs | Top-1 (%) | Top-5 (%) |
|:---|:---:|:---:|:---:|
| MAE (Baseline) | 100 | 59.87 | 93.48 |
| MInM (Tuned) | 100 | **60.28** | **93.55** |
| MInM (Long-horizon) | 400 | **69.38** | **95.75** |

> Long-horizon training reveals MInM's stronger training persistence: top-1 accuracy continues to improve steadily from 56.23% (epoch 100) to 69.38% (epoch 400).

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
│   ├── train_imagenette_mae.py  # MAE baseline on ImageNette
│   ├── imagenet_1kminm_parallel.py # Parallel MInM training on ImageNet-1K
│   ├── generate_sam_masks.py    # SAM instance mask generation
│   ├── main_pretrain.py     # MAE pre-training (ImageNet)
│   ├── main_finetune.py     # Fine-tuning script
│   ├── main_linprobe.py     # Linear probing script
│   └── submitit_*.py        # Distributed training wrappers
├── util/                    # Utilities (LR scheduling, LARS, etc.)
├── data/                    # Dataset storage
│   ├── imagenet/            # ImageNet-1K + SAM masks
│   └── imagenette/          # Imagenette + SAM masks
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
- [**SAM2**](https://github.com/facebookresearch/segment-anything-2): Segment Anything Model 2 for instance mask generation.
- [**DeiT**](https://github.com/facebookresearch/deit): Data-efficient Image Transformers.
- [**timm**](https://github.com/rwightman/pytorch-image-models): PyTorch Image Models.

## License

This project is licensed under the [CC-BY-NC 4.0 License](LICENSE).
