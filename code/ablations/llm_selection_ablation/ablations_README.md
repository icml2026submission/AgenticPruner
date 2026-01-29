# Ablation Study: LLM Selection

This directory contains configurations and instructions for reproducing the LLM selection ablation study from the paper.

---

## Overview

We evaluate three different LLM models for the Analysis Agent to determine their effectiveness in generating pruning strategies:

1. **Claude 3.5 Sonnet** (Our method) - Best performance
2. **GPT-4** - Competitive baseline
3. **Llama-3-70B** - Open-source alternative

**Key Finding:** Claude 3.5 Sonnet achieves the fastest convergence (3.2 average revisions) and highest accuracy (77.04%) compared to GPT-4 (4.1 revisions, 76.12% accuracy) and Llama-3-70B (5.6 revisions, 74.89% accuracy).

---

## Reproducing Results

### 1. Claude 3.5 Sonnet (Default - Our Method)

```bash
cd ../  # Return to code/ directory

python main.py \
    --model resnet50 \
    --dataset imagenet \
    --data_path /path/to/imagenet \
    --target_macs 2.06 \
    --llm_model claude-3-5-sonnet-20241022 \
    --llm_temperature 0 \
    --max_revisions 100 \
    --batch_size 256 \
    --epochs 100
```

**Expected Results:**
- Average revisions: ~3.2
- Convergence time: Baseline (1.0×)
- Final MACs: ~1.81G
- Final accuracy: ~77.04%

### 2. GPT-4

```bash
python main.py \
    --model resnet50 \
    --dataset imagenet \
    --data_path /path/to/imagenet \
    --target_macs 2.06 \
    --llm_model gpt-4 \
    --llm_temperature 0 \
    --max_revisions 100 \
    --batch_size 256 \
    --epochs 100
```

**Expected Results:**
- Average revisions: ~4.1
- Convergence time: 28% slower than Claude (1.26× baseline)
- Final MACs: ~1.72G
- Final accuracy: ~76.12%

### 3. Llama-3-70B

```bash
python main.py \
    --model resnet50 \
    --dataset imagenet \
    --data_path /path/to/imagenet \
    --target_macs 2.06 \
    --llm_model llama-3-70b \
    --llm_temperature 0 \
    --max_revisions 100 \
    --batch_size 256 \
    --epochs 100
```

**Expected Results:**
- Average revisions: ~5.6
- Convergence time: 67% slower than Claude (1.67× baseline)
- Final MACs: ~1.68G
- Final accuracy: ~74.89%

---

## Key Parameters

All experiments use identical parameters except for the LLM model:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--llm_model` | claude-3-5-sonnet-20241022 / gpt-4 / llama-3-70b | LLM selection |
| `--llm_temperature` | 0 | Deterministic generation for reproducibility |
| `--llm_max_tokens` | 1000 | Sufficient for strategy JSON |
| `--max_revisions` | 100 | Safety limit for convergence |
| `--target_macs` | 2.06 | MAC budget in GigaMACs |

---

## API Configuration

### For Claude 3.5 Sonnet (Default)
```bash
export OPENROUTER_API_KEY="your_openrouter_key"
# OR
export ANTHROPIC_API_KEY="your_anthropic_key"
```

### For GPT-4
```bash
export OPENAI_API_KEY="your_openai_key"
```

### For Llama-3-70B
```bash
export OPENROUTER_API_KEY="your_openrouter_key"
```

---

## Metrics Tracked

For each LLM, the following metrics are logged:

1. **Number of revisions** - How many LLM calls needed to converge
2. **Convergence time** - Total time to reach MAC target
3. **Final MACs** - Achieved computational budget
4. **Final accuracy** - Top-1 accuracy on ImageNet validation
5. **Strategy quality** - Coherence of generated pruning strategies

---

## Expected Results Summary

| LLM Model | Avg Revisions | Time vs Claude | Final Acc (%) | Final MACs (G) |
|-----------|---------------|----------------|---------------|----------------|
| Claude 3.5 Sonnet | 3.2 | 1.0× (baseline) | 77.04 | 1.81           |
| GPT-4 | 4.1 | 1.26× (28% slower) | 76.12 | 1.72          |
| Llama-3-70B | 5.6 | 1.67× (67% slower) | 74.89 | 1.68          |

**Key Insights:** 
- All LLMs eventually converge to similar MAC budgets, but Claude 3.5 Sonnet requires significantly fewer iterations (3.2 vs 4.1 for GPT-4 and 5.6 for Llama-3-70B).
- Claude achieves the best accuracy (77.04%), with GPT-4 trailing by 0.92% (76.12%) and Llama-3-70B by 2.15% (74.89%).
- Claude reduces total pruning time by 28% compared to GPT-4 and 67% compared to Llama-3-70B.
- Llama-3-70B struggles with complex multi-step reasoning, resulting in both more revisions and lower final accuracy.

---

## Notes

1. **API Costs**: Claude and GPT-4 are commercial APIs with per-token pricing. Llama-3-70B can be accessed via OpenRouter or self-hosted.

2. **Reproducibility**: With temperature=0, results should be deterministic. However, minor variations (±0.5% accuracy, ±0.1G MACs) may occur due to randomness in fine-tuning and data sampling.

3. **Convergence**: All models eventually reach the MAC target within the ±1%/−5% tolerance band. The key difference is in efficiency (number of revisions) and final accuracy achieved.

4. **Quality**: We observed Claude generates more structurally coherent JSON responses with fewer parsing errors compared to other LLMs, contributing to its faster convergence.
