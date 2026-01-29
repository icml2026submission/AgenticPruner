# AgenticPruner

Official code for **"AgenticPruner: MAC-Constrained Neural Network Compression via LLM-Driven Strategy Search"**.

---

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Set API Key

```bash
export ANTHROPIC_API_KEY="your_api_key"
# OR
export OPENROUTER_API_KEY="your_api_key"
```

### Prepare ImageNet-1K

```bash
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

---

## Reproducing Main Results

### ResNet-50 (Table 1)

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
    --epochs 100 \
    --output_dir ./outputs/resnet50
```

**Expected:** Converges in 3-5 revisions, ~1.81G MACs, ~77.04% accuracy

### DeiT-Tiny (Table 2)

```bash
python main.py \
    --model deit_tiny \
    --dataset imagenet \
    --data_path /path/to/imagenet \
    --target_macs 0.62 \
    --llm_model claude-3-5-sonnet-20241022 \
    --llm_temperature 0 \
    --optimizer adamw \
    --lr 0.0005 \
    --weight_decay 0.05 \
    --batch_size 2048 \
    --epochs 300 \
    --output_dir ./outputs/deit_tiny
```

**Expected:** ~2.90M params (from 5.9M), ~0.61G MACs, ~71.65% accuracy

### Other Models

For ResNet-101 and ConvNeXt-Small, use the same command structure with parameters from the table below.

---

## Key Arguments

```
--model              Model architecture (resnet50, resnet101, convnext_small, deit_tiny)
--target_macs        Target MAC budget in GigaMACs
--data_path          Path to ImageNet directory
--output_dir         Checkpoint save directory

--llm_model          LLM model (default: claude-3-5-sonnet-20241022)
--llm_temperature    LLM temperature (default: 0)
--importance         Importance criterion (default: taylor)

--optimizer          sgd (CNNs) or adamw (ViTs)
--lr                 Learning rate
--weight_decay       Weight decay
--batch_size         Training batch size
--epochs             Fine-tuning epochs
```

---

## Hyperparameter Summary

| Model | Target MACs | LR | Weight Decay | Optimizer | Batch | Epochs |
|-------|-------------|----|--------------| ----------|-------|--------|
| ResNet-50 | 2.06G | 0.08 | 0.0001 | SGD (mom=0.9) | 1024 | 100 |
| ResNet-101 | 4.48G | 0.08 | 0.0001 | SGD (mom=0.9) | 1024 | 100 |
| ConvNeXt-Small | 8.48G | 0.08 | 0.0001 | SGD (mom=0.9) | 1024 | 100 |
| DeiT-Tiny | 0.62G | 0.0005 | 0.05 | AdamW | 2048 | 300 |

All experiments use `claude-3-5-sonnet-20241022` with `temperature=0` and `taylor` importance.

---

## Ablation Studies

For LLM comparison and other ablations, see `ablations/README.md`.

**Example - LLM comparison:**
```bash
# GPT-4
python main.py --model resnet50 --target_macs 2.06 --llm_model gpt-4 --llm_temperature 0

# Llama-3-70B  
python main.py --model resnet50 --target_macs 2.06 --llm_model llama-3-70b --llm_temperature 0
```

---

## Evaluation

```bash
python eval_agent.py \
    --checkpoint ./outputs/resnet50/resnet50_finetuned.pth \
    --model resnet50 \
    --dataset imagenet \
    --data_path /path/to/imagenet \
    --batch_size 256
```

---

## Experiment Tracking

Experiments log to Weights & Biases (optional):

```bash
wandb login
python main.py --wandb_project AgenticPruner [other args]
```

Logged metrics: training curves, MAC convergence, LLM strategies, hardware performance.

---

## Reproducibility

- Random seed: 42 (set in all scripts)
- Deterministic LLM: temperature=0
- Pre-trained models: timm library
- Hardware requirements: GPU with ≥24GB VRAM, ≥64GB RAM
