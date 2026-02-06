# Venator: Results

> **Note:** This is a template. Run the full pipeline (via dashboard or CLI) to populate with actual numbers from your dataset and model.

## 1. Overall Metrics

| Metric | Ensemble | PCA+Mahalanobis | Isolation Forest | Autoencoder |
|--------|----------|-----------------|------------------|-------------|
| AUROC | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| AUPRC | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| FPR@95TPR | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Precision | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Recall | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| F1 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Model:** Mistral-7B-Instruct-v0.3 (4-bit quantized via MLX)
**Layer:** _TBD_ (selected via ablation or default middle layer)
**PCA dimensions:** 50
**Threshold:** 95th percentile of validation scores
**Training set:** _TBD_ benign prompts
**Test set:** _TBD_ benign + _TBD_ jailbreak prompts

## 2. Per-Detector Comparison

### AUROC by Detector

```
Detector             AUROC
─────────────────────────────
PCA + Mahalanobis    _TBD_
Isolation Forest     _TBD_
Autoencoder          _TBD_
Ensemble (weighted)  _TBD_
```

### Score Correlation Matrix

Low pairwise correlation between detectors confirms decorrelated errors, validating the ensemble approach per the ELK paper.

```
                     PCA+Maha    IForest    AutoEnc    Ensemble
PCA+Mahalanobis      1.00        _TBD_      _TBD_      _TBD_
Isolation Forest     _TBD_       1.00       _TBD_      _TBD_
Autoencoder          _TBD_       _TBD_      1.00       _TBD_
Ensemble             _TBD_       _TBD_      _TBD_      1.00
```

**Insight:** _Fill in after running detector comparison ablation. Expected: moderate correlation (0.4-0.7) between individual detectors, confirming complementary error profiles._

## 3. Ablation Results

### 3.1 Layer Selection

| Layer | AUROC | AUPRC | FPR@95TPR |
|-------|-------|-------|-----------|
| 8 | _TBD_ | _TBD_ | _TBD_ |
| 10 | _TBD_ | _TBD_ | _TBD_ |
| 12 | _TBD_ | _TBD_ | _TBD_ |
| 14 | _TBD_ | _TBD_ | _TBD_ |
| 16 | _TBD_ | _TBD_ | _TBD_ |
| 18 | _TBD_ | _TBD_ | _TBD_ |
| 20 | _TBD_ | _TBD_ | _TBD_ |
| 22 | _TBD_ | _TBD_ | _TBD_ |
| 24 | _TBD_ | _TBD_ | _TBD_ |

**Best layer:** _TBD_

**Insight:** _Expected pattern: middle layers (12-20) outperform early and late layers, consistent with the Earliest Informative Layer criterion from Mallen et al. (2024)._

### 3.2 PCA Dimensions

| Dimensions | AUROC | AUPRC | Sample/Feature Ratio |
|------------|-------|-------|----------------------|
| 10 | _TBD_ | _TBD_ | _TBD_ |
| 20 | _TBD_ | _TBD_ | _TBD_ |
| 30 | _TBD_ | _TBD_ | _TBD_ |
| 50 | _TBD_ | _TBD_ | _TBD_ |
| 75 | _TBD_ | _TBD_ | _TBD_ |
| 100 | _TBD_ | _TBD_ | _TBD_ |

**Best dimensions:** _TBD_

**Insight:** _Expected: diminishing returns beyond 30-50 dimensions. Too few lose signal; too many degrade the sample-to-feature ratio._

### 3.3 Detector Comparison

| Configuration | AUROC | AUPRC | Training Time |
|--------------|-------|-------|---------------|
| PCA+Mahalanobis only | _TBD_ | _TBD_ | _TBD_ |
| Isolation Forest only | _TBD_ | _TBD_ | _TBD_ |
| Autoencoder only | _TBD_ | _TBD_ | _TBD_ |
| Full ensemble | _TBD_ | _TBD_ | _TBD_ |

**Insight:** _Expected: the ensemble outperforms individual detectors, with the improvement attributable to decorrelated errors._

## 4. Example Detections

### High-Confidence True Positives

Jailbreak prompts correctly detected with the highest ensemble scores:

| Score | Prompt (truncated) | Type |
|-------|-------------------|------|
| _TBD_ | _TBD_ | _TBD_ |
| _TBD_ | _TBD_ | _TBD_ |
| _TBD_ | _TBD_ | _TBD_ |

### Correctly Classified Benign Prompts

Normal prompts with the lowest scores (high confidence "normal"):

| Score | Prompt (truncated) |
|-------|-------------------|
| _TBD_ | _TBD_ |
| _TBD_ | _TBD_ |
| _TBD_ | _TBD_ |

## 5. Failure Analysis

### False Positives (Benign Flagged as Anomaly)

| Score | Prompt (truncated) | Likely Cause |
|-------|-------------------|--------------|
| _TBD_ | _TBD_ | _TBD_ |
| _TBD_ | _TBD_ | _TBD_ |

**Common FP patterns:** _Fill in after evaluation. Typical causes: unusual formatting, code-heavy prompts, or topics underrepresented in training data._

### False Negatives (Jailbreaks Missed)

| Score | Prompt (truncated) | Attack Type |
|-------|-------------------|-------------|
| _TBD_ | _TBD_ | _TBD_ |
| _TBD_ | _TBD_ | _TBD_ |

**Common FN patterns:** _Fill in after evaluation. Typical causes: subtle jailbreaks that closely mimic benign prompt structure, or attacks that operate through the model's normal processing pathways._

## 6. Discussion

### Strengths

- **Novel attack detection:** The unsupervised approach detects attack types not present in any training set, by definition.
- **No labeled jailbreaks required for training:** Removes the dependency on curating and maintaining jailbreak datasets.
- **Interpretable scores:** Per-detector breakdown shows which aspect of "normality" was violated.
- **Fast inference:** Scoring a single prompt takes ~1-3 seconds (dominated by MLX forward pass).

### Limitations

- **Requires access to model internals:** Only works when you can extract hidden states (white-box setting).
- **Threshold sensitivity:** The 95th percentile threshold is a design choice — more conservative thresholds reduce false positives but may miss subtle attacks.
- **Distribution shift:** If the benign prompt distribution shifts significantly (e.g., new topic domains), the model may need retraining.
- **Apple Silicon dependency:** MLX is required for efficient local inference, limiting portability.

### Future Work

- **Multi-layer ensembling:** Combine signals from multiple layers rather than selecting a single best layer.
- **Online learning:** Update the benign distribution incrementally as new prompts are processed.
- **Cross-model transfer:** Test whether detectors trained on one model's activations transfer to another.
- **Supervised-unsupervised fusion:** Combine Venator's anomaly scores with supervised classifier confidence for a more robust system.
- **Larger-scale evaluation:** Test on broader jailbreak datasets (GCG, AutoDAN, multilingual attacks).

## 7. Reproducibility

All results can be reproduced by running:

```bash
# Dashboard (interactive)
streamlit run venator/dashboard/app.py

# CLI (scripted)
python scripts/collect_prompts.py --output data/prompts/
python scripts/extract_activations.py --prompts data/prompts/benign.jsonl --output data/activations/store.h5
python scripts/create_splits.py --store data/activations/store.h5 --output data/
python scripts/train_detector.py --store data/activations/store.h5 --splits data/splits.json --output models/detector_v1/
python scripts/evaluate.py --store data/activations/store.h5 --splits data/splits.json --model-dir models/detector_v1/
python scripts/run_ablations.py --store data/activations/store.h5 --splits data/splits.json --output results/ablations/
```

Random seed: 42 (configurable via `VENATOR_RANDOM_SEED`)
