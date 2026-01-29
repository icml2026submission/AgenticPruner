# ICML 2026 Supplementary Materials

Anonymous submission for **"AgenticPruner: MAC-Constrained Neural Network Compression via LLM-Driven Strategy Search"**.

---

## 📦 Repository Structure

This submission consists of two components:

### 1. **Code Repository (GitHub - This Repository)**
Contains complete implementation, scripts, and documentation.

### 2. **Model Checkpoints (Hugging Face)**
🔗 **https://huggingface.co/AgenticPruner/checkpoints**

All trained model weights are hosted on Hugging Face due to file size constraints.

**⚠️ Setup Required:**
1. Clone this GitHub repository for code
2. Download checkpoint files from the Hugging Face link above
3. Place downloaded checkpoints in a `checkpoints/` directory at the repository root

---

## Contents

This package provides complete implementation code, complete documentation and trained models for reproducing all experimental results.

**Structure:**
```
GitHub Repository (this repository):
├── README.md                                  # This file
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

Hugging Face Repository (external):
https://huggingface.co/AgenticPruner/checkpoints
└── Download these checkpoint files:
    ├── README.md                             # Documentation for all checkpoints
    ├── resnet50_finetuned_clean.pth          # 1.81G MACs, 77.04% accuracy
    ├── resnet50_pruned_clean.pt       
    ├── resnet101_finetuned_clean.pth         # 4.17G MACs, 78.9% accuracy
    ├── resnet101_pruned_clean.pt
    ├── convnext_small_finetuned_clean.pth    # 8.17G MACs, 82.27% accuracy
    ├── convnext_small_pruned_clean.pt
    ├── deittiny_finetuned_clean.pth          # 0.61G MACs, 71.65% accuracy
    └── deittiny_pruned_clean.pt       
```

---

## Quick Start

### 1. Setup Repository and Environment
```bash
# Clone this repository
git clone [anonymous-github-link]
cd [repo-name]

# Create conda environment
conda create -n agenticpruner python=3.10 -y
conda activate agenticpruner

# Install dependencies
cd code
pip install -r requirements.txt
```

### 2. Download Model Checkpoints
```bash
# Go back to repository root
cd ..

# Create checkpoints directory
mkdir -p checkpoints

# Visit: https://huggingface.co/AgenticPruner/checkpoints
# Click "Files and versions" tab
# Download the checkpoint files you need
# Place them in the checkpoints/ directory
#
# Alternatively, use huggingface-cli:
cd checkpoints
huggingface-cli download AgenticPruner/checkpoints --local-dir .
cd ..
```

### 3. API Configuration
```bash
export ANTHROPIC_API_KEY="your_api_key"
# OR
export OPENROUTER_API_KEY="your_api_key"
```

### 4. Run Pruning (ResNet-50)
```bash
cd code
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

### 5. Evaluate Provided Checkpoints
```bash
# Make sure checkpoints are downloaded to ../checkpoints/

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
    --checkpoint ../../checkpoints/resnet50_pruned_clean.pt \
    --data_path /path/to/imagenet \
    --epochs 100

# Evaluate model
python evaluate.py \
    --model resnet50 \
    --checkpoint ../../checkpoints/resnet50_finetuned_clean.pth \
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

**🔗 Download from Hugging Face:** https://huggingface.co/AgenticPruner/checkpoints

We provide both pruned and fine-tuned checkpoints for all four models:

| Model | Pruned Checkpoint | Fine-tuned Checkpoint | MACs | Accuracy |
|-------|-------------------|----------------------|------|----------|
| ResNet-50 | `resnet50_pruned_clean.pt` | `resnet50_finetuned_clean.pth` | 1.81G | 77.04%   |
| ResNet-101 | `resnet101_pruned_clean.pt` | `resnet101_finetuned_clean.pth` | 4.17G | 78.9%    |
| ConvNeXt-Small | `convnext_small_pruned_clean.pt` | `convnext_small_finetuned_clean.pth` | 8.17G | 82.27%   |
| DeiT-Tiny | `deittiny_pruned_clean.pt` | `deittiny_finetuned_clean.pth` | 0.61G | 71.65%   |

**Download Instructions:**
1. Visit https://huggingface.co/AgenticPruner/checkpoints
2. Click "Files and versions" tab
3. Download required `.pth` and `.pt` files
4. Place them in `checkpoints/` directory at repository root
5. Verify paths match the evaluation commands above

See `checkpoints/README.md` (included in huggingface repository) for detailed loading instructions.

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
