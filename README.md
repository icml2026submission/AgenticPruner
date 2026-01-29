# ICML 2026 Supplementary Materials

Anonymous submission for **"AgenticPruner: MAC-Constrained Neural Network Compression via LLM-Driven Strategy Search"**.

---

## Contents

This package contains implementation code, model checkpoints, and complete documentation for reproducing all experimental results.

**Structure:**
```
supplementary/
├── README.md                                  # This file
├── checkpoints/                               # Model checkpoints
│   ├── CHECKPOINTS_README.md          
│   ├── resnet50_finetuned_clean.pth          # 1.81G MACs, 77.04% accuracy
│   ├── resnet50_pruned_clean.pt       
│   ├── resnet101_finetuned_clean.pth         # 4.17G MACs, 78.9% accuracy
│   ├── resnet101_pruned_clean.pt
│   ├── convnext_small_finetuned_clean.pth    # 8.17G MACs, 82.27% accuracy
│   ├── convnext_small_pruned_clean.pt
│   ├── deittiny_finetuned_clean.pth          # 0.61G MACs, 71.65% accuracy
│   └── deittiny_pruned_clean.pt       
└── code/                                      # Complete source code
    ├── README.md                              # Code documentation
    ├── requirements.txt                       # Exact dependencies
    ├── main.py                                # Main entry point
    ├── workflow.py                            # Multi-agent orchestration
    ├── *_agent.py                             # Agent implementations
    ├── deepseek_llm.py                        # LLM API integration
    ├── data/                                  # Dataset loaders
    ├── llm/                                   # LLM prompts and parsing
    ├── utils/                                 # Utility functions
    ├── scripts/                               # Standalone scripts
    ├── ablations/                             # Ablation configurations
    └── pbench/                                # Benchmarking tools
```

---

## Quick Start

### Environment Setup
```bash
conda create -n agenticpruner python=3.10 -y
conda activate agenticpruner
cd code
pip install -r requirements.txt
```

### API Configuration
```bash
export ANTHROPIC_API_KEY="your_api_key"
# OR
export OPENROUTER_API_KEY="your_api_key"
```

### Run Pruning (ResNet-50)
```bash
python main.py \
    --model resnet50 \
    --dataset imagenet \
    --data_path /path/to/imagenet \
    --target_macs 2.06 \
    --llm_model claude-3-5-sonnet-20241022 \
    --llm_temperature 0 \
    --optimizer sgd \
    --lr 0.08 \
    --momentum 0.9 \
    --weight_decay 0.0001 \
    --batch_size 1024 \
    --epochs 100
```

### Evaluate Provided Checkpoints
```bash
# ResNet-50
python eval_agent.py \
    --checkpoint ../checkpoints/resnet50_finetuned_clean.pth \
    --model resnet50 \
    --dataset imagenet \
    --data_path /path/to/imagenet \
    --batch_size 256

# ResNet-101
python eval_agent.py \
    --checkpoint ../checkpoints/resnet101_finetuned_clean.pth \
    --model resnet101 \
    --dataset imagenet \
    --data_path /path/to/imagenet \
    --batch_size 256

# ConvNeXt-Small
python eval_agent.py \
    --checkpoint ../checkpoints/convnext_small_finetuned_clean.pth \
    --model convnext_small \
    --dataset imagenet \
    --data_path /path/to/imagenet \
    --batch_size 256

# DeiT-Tiny
python eval_agent.py \
    --checkpoint ../checkpoints/deittiny_finetuned_clean.pth \
    --model deit_tiny \
    --dataset imagenet \
    --data_path /path/to/imagenet \
    --batch_size 256
```

### Standalone Scripts
```bash
cd code/scripts

# Fine-tune a pruned model
python finetuning.py \
    --model resnet50 \
    --checkpoint ../checkpoints/resnet50_pruned_clean.pt \
    --data_path /path/to/imagenet \
    --epochs 100

# Evaluate model
python evaluate.py \
    --model resnet50 \
    --checkpoint ../checkpoints/resnet50_finetuned_clean.pth \
    --data_path /path/to/imagenet
```

---

## Reproducing Paper Results

### Main Results (Tables 1 & 2)

| Model | Baseline MACs | Target MACs | Command |
|-------|---------------|-------------|---------|
| ResNet-50 | 4.1G | 2.06G | `python main.py --model resnet50 --target_macs 2.06 --dataset imagenet --data_path /path/to/imagenet` |
| ResNet-101 | 7.8G | 4.48G | `python main.py --model resnet101 --target_macs 4.48 --dataset imagenet --data_path /path/to/imagenet` |
| ConvNeXt-Small | 8.7G | 8.48G | `python main.py --model convnext_small --target_macs 8.48 --dataset imagenet --data_path /path/to/imagenet` |
| DeiT-Tiny | 1.3G | 0.62G | `python main.py --model deit_tiny --target_macs 0.62 --dataset imagenet --data_path /path/to/imagenet` |

**Expected Results:**
- **ResNet-50**: ~1.81G MACs, ~77.04% accuracy (+0.91% vs 76.13% baseline)
- **ResNet-101**: ~4.17G MACs, ~78.9% accuracy (+1.52% vs 77.38% baseline)
- **ConvNeXt-Small**: ~8.17G MACs, ~82.27% accuracy (-1.56% vs 83.83% baseline)
- **DeiT-Tiny**: ~0.61G MACs, ~71.65% accuracy, 2.90M params (from 5.9M baseline)

For complete hyperparameters, see `code/README.md`.

### Ablation Studies

See `code/ablations/README.md` for LLM comparison and other ablation experiments.

---

## Provided Checkpoints

We provide both pruned and fine-tuned checkpoints for all four models:

| Model | Pruned Checkpoint | Fine-tuned Checkpoint | MACs | Accuracy |
|-------|-------------------|----------------------|------|----------|
| ResNet-50 | `resnet50_pruned_clean.pt` | `resnet50_finetuned_clean.pth` | 1.81G | 77.04%   |
| ResNet-101 | `resnet101_pruned_clean.pt` | `resnet101_finetuned_clean.pth` | 4.17G | 78.9%    |
| ConvNeXt-Small | `convnext_small_pruned_clean.pt` | `convnext_small_finetuned_clean.pth` | 8.17G | 82.27%   |
| DeiT-Tiny | `deittiny_pruned_clean.pt` | `deittiny_finetuned_clean.pth` | 0.61G | 71.65%   |

See `checkpoints/CHECKPOINTS_README.md` for detailed loading instructions and metadata.

---

## Documentation

- **Code**: `code/README.md` - Complete usage guide and hyperparameters
- **Checkpoints**: `checkpoints/CHECKPOINTS_README.md` - Model loading instructions
- **Ablations**: `code/ablations/README.md` - Ablation study reproduction

---

## Requirements

**Hardware:**
- GPU with ≥24GB VRAM
- ≥64GB RAM
- ~200GB storage for ImageNet + checkpoints

**Software:**
- Python 3.10+
- PyTorch 2.5.1, torchvision 0.20.1
- torch-pruning 1.6.0, timm 1.0.19
- See `code/requirements.txt` for complete list

---

## Reproducibility

All experiments use:
- Random seed: 42
- Deterministic LLM: temperature=0
- Pre-trained models from timm library
- Exact package versions in `requirements.txt`
