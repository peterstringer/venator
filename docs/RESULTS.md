# Venator: Results

## 1. Overall Metrics

**Model:** Mistral-7B-Instruct-v0.3 (4-bit quantized via MLX)
**Primary layer:** 18 (optimal from ablation; layers 10–24 all within 0.001 AUROC)
**PCA dimensions:** 50
**Threshold:** Youden's J statistic on labeled validation data (0.550)
**Test set:** 150 benign + 350 jailbreak prompts (held-out, never seen during training)

### Primary Detector: Linear Probe

| Metric | Value |
|--------|-------|
| AUROC | 0.999 |
| AUPRC | 1.000 |
| F1 | 0.991 |
| Precision | 0.997 |
| Recall | 0.986 |
| FPR | 0.007 |

## 2. Full Detector Comparison

| Detector | Type | AUROC | AUPRC | F1 | Precision | Recall | FPR |
|----------|------|-------|-------|-----|-----------|--------|-----|
| **Linear Probe** | **Supervised** | **0.999** | **1.000** | **0.991** | **0.997** | **0.986** | **0.007** |
| MLP Probe | Supervised | 0.997 | 0.999 | 0.990 | 0.994 | 0.986 | 0.013 |
| Contrastive Mahalanobis | Supervised | 0.995 | 0.998 | 0.975 | 0.991 | 0.960 | 0.020 |
| Custom Ensemble | Ensemble | 0.967 | 0.980 | 0.970 | 0.941 | 1.000 | 0.147 |
| Autoencoder | Unsupervised | 0.869 | 0.924 | 0.878 | 0.880 | 0.877 | 0.280 |
| PCA + Mahalanobis | Unsupervised | 0.737 | 0.818 | 0.823 | 0.811 | 0.834 | 0.453 |
| Isolation Forest | Unsupervised | 0.711 | 0.799 | 0.787 | 0.827 | 0.751 | 0.367 |
| Contrastive Direction | Supervised | 0.656 | 0.729 | 0.880 | 0.802 | 0.974 | 0.560 |

**Key insight:** The linear probe dominates. The ensemble is strictly worse (0.967 vs 0.999) — weaker detectors add noise. Even the MLP probe's extra capacity provides no benefit (0.997 vs 0.999). The jailbreak boundary in activation space is linear.

## 3. Ablation Results

### 3.1 Layer Selection (Linear Probe)

| Layer | AUROC | AUPRC | FPR@95TPR |
|-------|-------|-------|-----------|
| 4 | 0.981 | 0.993 | 0.087 |
| 6 | 0.991 | 0.997 | 0.020 |
| 8 | 0.997 | 0.999 | 0.007 |
| 10 | 0.998 | 0.999 | 0.000 |
| 12 | 0.999 | 1.000 | 0.000 |
| 14 | 0.999 | 1.000 | 0.000 |
| 16 | 1.000 | 1.000 | 0.000 |
| **18** | **1.000** | **1.000** | **0.000** |
| 20 | 1.000 | 1.000 | 0.000 |
| 22 | 1.000 | 1.000 | 0.000 |
| 24 | 1.000 | 1.000 | 0.000 |

**Best layer:** 18 (marginal advantage; layers 10–24 are functionally equivalent)

**Insight:** The jailbreak signal emerges early (layer 4, 0.981 AUROC) and saturates by layer 10 (0.998+). This confirms the Earliest Informative Layer criterion — middle layers provide strong signal with the broadest generalization. The signal remains strong through later layers, unlike some prior ELK results that showed degradation.

### 3.2 Layer Selection (MLP Probe)

| Layer | AUROC | AUPRC | FPR@95TPR |
|-------|-------|-------|-----------|
| 4 | 0.992 | 0.997 | 0.033 |
| 6 | 0.994 | 0.998 | 0.013 |
| 8 | 0.994 | 0.998 | 0.007 |
| 10 | 0.998 | 0.999 | 0.000 |
| 12 | 0.998 | 0.999 | 0.000 |
| 14 | 0.996 | 0.999 | 0.007 |
| 16 | 0.998 | 0.999 | 0.000 |
| 18 | 0.997 | 0.999 | 0.000 |
| 20 | 0.998 | 0.999 | 0.000 |
| 22 | 0.998 | 0.999 | 0.000 |
| **24** | **0.999** | **1.000** | **0.000** |

**Insight:** The MLP probe shows slightly more variance across layers than the linear probe, consistent with its higher capacity leading to less stable optimization.

### 3.3 PCA Dimensions (Linear Probe)

| Dimensions | AUROC | AUPRC | FPR@95TPR |
|------------|-------|-------|-----------|
| None (4096) | 0.9997 | 1.000 | 0.000 |
| 10 | 0.9964 | 0.999 | 0.007 |
| 20 | 0.9993 | 1.000 | 0.000 |
| 30 | 0.9993 | 1.000 | 0.000 |
| **50** | **0.9995** | **1.000** | **0.000** |
| 75 | 0.9997 | 1.000 | 0.000 |
| 100 | 0.9997 | 1.000 | 0.000 |
| 150 | 0.9997 | 1.000 | 0.000 |
| 200 | 0.9997 | 1.000 | 0.000 |

**Insight:** Remarkably insensitive to PCA dimensionality. Even 20 dimensions retain 99.93% of the discrimination signal from raw 4096-dimensional activations. PCA=50 (default) is near-optimal.

### 3.4 PCA Dimensions (MLP Probe)

| Dimensions | AUROC |
|------------|-------|
| None (4096) | 0.9957 |
| 10 | 0.9969 |
| **20** | **0.9996** |
| 30 | 0.9975 |
| 50 | 0.9985 |
| 75 | 0.9966 |
| 100 | 0.9964 |

**Insight:** The MLP probe is more sensitive to PCA dimensionality than the linear probe. It peaks at PCA=20 (0.9996) and degrades at higher dimensions, suggesting overfitting when given more features with the small training set.

### 3.5 Label Efficiency

| Labeled Jailbreaks | Linear Probe AUROC | MLP Probe AUROC |
|--------------------|-------------------|-----------------|
| 5 | 0.996 | 0.992 |
| 10 | 0.996 | 0.994 |
| 15 | 0.996 | 0.993 |
| 20 | 0.997 | 0.994 |
| 30 | 0.998 | 0.995 |
| 50 | 0.999 | 0.999 |
| 75 | 1.000 | 0.998 |

**Unsupervised baselines:**
- PCA + Mahalanobis: 0.638 AUROC
- Autoencoder: 0.794 AUROC (linear probe), 0.832 AUROC (MLP probe run)

**Insight:** Even 5 labeled jailbreaks push the linear probe well above any unsupervised method (0.996 vs 0.794 best unsupervised). The performance curve is nearly flat from 5 to 30 labels, suggesting the jailbreak direction is identified by the very first few examples. Returns plateau sharply after 30–50 labels.

### 3.6 Cross-Source Generalization

**Linear Probe:**

| Train ↓ Test → | Group A | Group B | Group C |
|----------------|---------|---------|---------|
| Group A | 0.999 | 0.999 | 0.998 |
| Group B | 0.999 | 0.999 | 0.998 |
| Group C | 0.998 | 0.997 | 0.996 |

**MLP Probe:**

| Train ↓ Test → | Group A | Group B | Group C |
|----------------|---------|---------|---------|
| Group A | 0.998 | 0.994 | 0.999 |
| Group B | 0.849 | 0.844 | 0.856 |
| Group C | 0.996 | 0.993 | 0.996 |

**Insight:** The linear probe generalizes near-perfectly across all jailbreak source combinations (0.996–0.999), suggesting a universal "jailbreak direction" in activation space. The MLP probe shows much weaker generalization — training on Group B drops to ~0.85 on all test groups, indicating the MLP overfits to source-specific patterns that don't transfer.

## 4. Score Distribution

The linear probe produces well-separated score distributions:
- **Benign prompts**: concentrated near 0 (mean ~0.06, most below 0.1)
- **Jailbreak prompts**: concentrated near 1 (mean ~0.94, most above 0.8)
- **Threshold**: 0.550 (Youden's J optimal)
- Only 1 benign prompt (0.7%) scores above the threshold (false positive)
- 5 jailbreak prompts (~1.4%) score below the threshold (false negatives)

The autoencoder, by contrast, shows heavily overlapping distributions with a threshold of 0.003, explaining its much lower AUROC (0.869).

## 5. Reproducibility

All results can be reproduced by running:

```bash
# Dashboard (interactive — Quick Run page)
streamlit run venator/dashboard/app.py

# CLI (scripted)
python scripts/collect_prompts.py --output data/prompts/
python scripts/extract_activations.py --prompts data/prompts/benign.jsonl --output data/activations/all.h5 --layers 4 6 8 10 12 14 16 18 20 22 24
python scripts/create_splits.py --store data/activations/all.h5 --output data/splits.json
python scripts/train_detector.py --store data/activations/all.h5 --splits data/splits.json --ensemble-type hybrid --layer 18 --output models/v2/
python scripts/evaluate_final.py --model-dir models/v2/ --store data/activations/all.h5 --splits data/splits.json --output results/final/
python scripts/run_supervised_ablations.py --store data/activations/all.h5 --splits data/splits.json --ablate label_budget generalization --output results/supervised_ablations/
```

Random seed: 42 (configurable via `VENATOR_RANDOM_SEED`)
