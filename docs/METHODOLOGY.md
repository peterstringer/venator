# Venator: Methodology

This document explains the research foundation, design decisions, and methodology behind Venator's approach to jailbreak detection via activation probing.

## 1. Background: The ELK Problem and Cheap Monitors

**Eliciting Latent Knowledge (ELK)** asks: can we extract what an AI model "knows" from its internal representations, even when its outputs might be unreliable?

Mallen et al. (2024) showed that linear probes on middle-layer activations can detect when language models behave anomalously — with 0.94+ AUROC — by learning what "normal" computation looks like. Burns et al. (2022) demonstrated that unsupervised methods can discover latent knowledge without labeled data by exploiting the structure of internal representations.

Anthropic's "Cheap Monitors" paper (Cunningham et al., 2025) extended this to safety: linear probes on intermediate model representations can match the performance of much larger dedicated classifiers at a fraction of the cost. This validated the core approach — reusing the representations the LLM already computes rather than training a separate classifier.

Venator applies these findings to jailbreak detection: a logistic regression on middle-layer activations achieves **0.999 AUROC** with only **30 labeled jailbreak examples**.

## 2. Core Hypothesis

> Jailbreak attempts produce statistically distinguishable activation patterns in LLM hidden states compared to benign prompts. A linear boundary in activation space is sufficient to separate them.

**Why this works:**
- LLM activations encode rich semantic representations of input intent
- Jailbreaks activate distinct computational pathways vs. benign prompts
- Even with few labeled examples, the activation space is informative
- The jailbreak "direction" in activation space is low-dimensional — a small number of examples suffice to identify it

## 3. Why Middle-Layer Activations?

Transformer models process input through a stack of layers. Different layers encode different types of information:

```
Layer 0-4 (early):     Token-level features, syntax, surface patterns
Layer 5-11 (early-mid): Semantic composition, entity recognition
Layer 12-20 (middle):   Behavioral intent, task representation  ← Venator targets these
Layer 21-28 (late-mid): Output preparation, generation planning
Layer 29-31 (final):    Next-token logit computation
```

**Why not embeddings (layer 0)?** Input embeddings reflect surface-level token features. A base64-encoded jailbreak looks entirely different from a plaintext one at the token level, but the model's middle layers converge to similar representations as it "understands" the decoded intent.

**Why not logits (final layer)?** The final layer is optimized for next-token prediction. A successful jailbreak produces *compliant* outputs — meaning the logits look normal even when the model is behaving dangerously. Middle layers capture the *process* of deciding to comply, not just the output.

**The Earliest Informative Layer criterion** (Mallen et al.): Middle layers provide the best trade-off between signal strength and generalization. Deeper layers overfit to specific output patterns; shallower layers lack the semantic abstraction needed to distinguish intent.

Venator extracts activations from layers 4–24 (even-numbered, configurable) for Mistral-7B (32 layers total). The ablation studies empirically validate which layer performs best. Results show:
- Signal emerges as early as layer 4 (AUROC 0.981 for linear probe)
- Performance plateaus around layer 10–12 (AUROC 0.998+)
- Layers 14–24 are near-identical (AUROC 0.9995) — the signal saturates
- No significant degradation in later layers (unlike some prior work)

**Optimal layer: 18** (by marginal AUROC for linear probe), though layers 10–24 are all within 0.001 AUROC of each other.

## 4. Why PCA Before Detection?

Mistral-7B has a hidden dimension of 4096. With limited training samples, high-dimensional methods suffer from poor covariance estimation and the curse of dimensionality.

**The problem:** Methods like Mahalanobis distance require inverting the covariance matrix. With more features than samples, the covariance matrix is singular. Even with regularization, the resulting metric is unreliable.

**The solution:** PCA reduces 4096 dimensions to a manageable number, giving a healthy sample-to-feature ratio. PCA also removes noise dimensions that don't contribute to the signal.

**Ablation results (linear probe):**

| PCA Dims | AUROC | Note |
|----------|-------|------|
| None (raw 4096) | 0.9997 | Best absolute, but slow |
| 10 | 0.9964 | Slight signal loss |
| 20 | 0.9993 | Good balance |
| 50 | 0.9995 | Default — near-optimal |
| 75 | 0.9997 | Matches raw |
| 100+ | 0.9997 | Diminishing returns |

**Key finding:** The linear probe is remarkably insensitive to PCA dimensionality. Even 20 dimensions retain nearly all the signal. The default of 50 provides a good balance between speed and accuracy.

**Note:** The MLP probe is more sensitive to PCA dimensionality — its best performance (0.9996 AUROC) occurs at PCA=20, with degradation at higher dimensions due to overfitting on the small training set.

## 5. Detection Architecture

### Primary Detector: Linear Probe (Logistic Regression)

The primary detector is a logistic regression on PCA-reduced activations. Score = P(jailbreak | activation).

**Pipeline:**
1. (Optional) PCA reduction on combined training activations
2. Logistic regression with balanced class weights
3. Score = predicted probability of jailbreak class

**Why logistic regression?**
- Minimal capacity prevents overfitting on small training sets
- Interpretable: the weight vector defines the "jailbreak direction" in activation space
- Fast: training takes ~0.1s, inference is a single matrix multiply
- Validated by Anthropic's Cheap Monitors paper

**Results (layer 18, PCA 50):**

| Metric | Value |
|--------|-------|
| AUROC | 0.999 |
| AUPRC | 1.000 |
| F1 | 0.991 |
| Precision | 0.997 |
| Recall | 0.986 |
| FPR | 0.007 |

### Comparison Detectors

All other detectors exist for research comparison. They quantify how much each architectural choice contributes:

| Detector | Type | AUROC | Role |
|----------|------|-------|------|
| **Linear Probe** | Supervised | **0.999** | **Primary** |
| MLP Probe | Supervised | 0.997 | More capacity, comparable accuracy |
| Contrastive Mahalanobis | Supervised | 0.995 | Class-conditional Mahalanobis |
| Custom Ensemble | Ensemble | 0.967 | Retired — hurts performance |
| Autoencoder | Unsupervised | 0.869 | Best unsupervised baseline |
| PCA + Mahalanobis | Unsupervised | 0.737 | Unsupervised baseline |
| Isolation Forest | Unsupervised | 0.711 | Unsupervised baseline |
| Contrastive Direction | Supervised | 0.656 | Diff-in-means (simplest supervised) |

### MLP Probe

A 2-layer MLP (input → 128 → 32 → 1) trained with BCE loss and early stopping. Sits between the linear probe (minimal capacity) and the autoencoder (unsupervised) in complexity.

**Finding:** The MLP probe achieves 0.997 AUROC — comparable to the linear probe but not better. The extra capacity doesn't help because the jailbreak boundary in activation space is approximately linear. The MLP is also more sensitive to PCA dimensionality and less stable across random seeds.

### Why No Ensemble

Empirical results confirm that ensembling hurts performance when one detector dominates:

- Linear Probe alone: **0.999 AUROC**
- Custom Ensemble (all detectors): **0.967 AUROC**

The ensemble drags performance down by ~3 percentage points. This matches the theoretical expectation: ensembles help when component errors are decorrelated AND components are comparably accurate. When one component is far stronger, the weaker detectors add noise.

The linear probe is used alone as the production detector. The ensemble infrastructure is retained for training (it routes data to supervised vs. unsupervised detectors correctly) but its weighted-average scoring is retired.

## 6. Unified Split Methodology

Venator uses a unified split that serves every detector type:

```
Split once, train anything.

Benign prompts:    70% train / 15% val / 15% test
Jailbreak prompts: 15% train / 15% val / 70% test

                    train_benign       → Unsupervised detector training
                    train_jailbreak    → Supervised detector training
                    val_benign         → Threshold calibration (all types)
                    val_jailbreak      → Threshold calibration (supervised)
                    test_benign        → Final evaluation only
                    test_jailbreak     → Final evaluation only
```

**How each detector type uses the split:**
- **Unsupervised**: `fit(train_benign)` → calibrate on `val_benign` → evaluate on `test_*`
- **Supervised**: `fit(train_benign + train_jailbreak)` → calibrate on `val_*` → evaluate on `test_*`
- **Ensemble**: each component uses its own path above → combine → evaluate on `test_*`

**Key constraint:** At least 50% of jailbreaks are reserved for uncontaminated testing (enforced programmatically). With default fractions, 70% of jailbreaks are held out — the supervised detectors never see them during training.

### Validation Checks

The `SplitManager.validate_splits()` programmatically enforces:
- Benign splits contain only benign labels
- Jailbreak splits contain only jailbreak labels
- No index appears in multiple splits
- All store indices are covered exactly once

## 7. Threshold Calibration

When labeled validation data is available, the primary detector uses **Youden's J statistic** for optimal threshold selection (maximizes TPR - FPR on the validation ROC curve).

**Fallback methods:**
- **Youden's J** (default with labeled val data): maximizes TPR - FPR
- **F1-optimal**: maximizes F1 score at the chosen threshold
- **FPR target**: finds the threshold matching a target false positive rate
- **Percentile** (default without labeled val data): sets threshold at the Nth percentile of benign validation scores

## 8. Label Efficiency

The ablation studies reveal that supervised detectors reach useful accuracy with surprisingly few labeled jailbreaks:

| Labeled Jailbreaks | Linear Probe AUROC | MLP Probe AUROC |
|--------------------|-------------------|-----------------|
| 5 | 0.996 | 0.992 |
| 10 | 0.996 | 0.994 |
| 15 | 0.996 | 0.993 |
| 20 | 0.997 | 0.994 |
| 30 | 0.998 | 0.995 |
| 50 | 0.999 | 0.999 |
| 75 | 1.000 | 0.998 |

**Unsupervised baselines for comparison:**
- PCA + Mahalanobis: 0.638 AUROC
- Autoencoder: 0.794 AUROC

**Key finding:** Even 5 labeled jailbreaks push the linear probe to 0.996 AUROC — far above any unsupervised method. Performance gains plateau around 30–50 examples. This aligns with the "cheap monitors" finding: the jailbreak direction in activation space is low-dimensional.

## 9. Cross-Source Generalization

Training on one jailbreak source and testing on others:

**Linear Probe (AUROC matrix):**

| Train → Test | Group A | Group B | Group C |
|-------------|---------|---------|---------|
| Group A | 0.999 | 0.999 | 0.998 |
| Group B | 0.999 | 0.999 | 0.998 |
| Group C | 0.998 | 0.997 | 0.996 |

**Finding:** The linear probe generalizes almost perfectly across jailbreak sources. This suggests a universal "jailbreak direction" in activation space — training on any category transfers to others.

**MLP Probe shows weaker generalization** — Group B as training source drops to ~0.85 AUROC on other groups, indicating the MLP overfits to source-specific patterns.

## 10. Evaluation Metrics

### Primary: AUROC

**Area Under the ROC Curve.** Measures the detector's ability to rank jailbreaks higher than benign prompts across all possible thresholds. Threshold-independent, making it the fairest comparison metric.

### Secondary Metrics

| Metric | What it measures | Reported at |
|--------|-----------------|-------------|
| AUPRC | Area Under Precision-Recall curve | All thresholds |
| FPR@95TPR | False positive rate when 95% of jailbreaks are caught | Fixed TPR = 0.95 |
| Precision | Of flagged prompts, how many are actually jailbreaks? | Chosen threshold |
| Recall | Of all jailbreaks, how many were flagged? | Chosen threshold |
| F1 | Harmonic mean of precision and recall | Chosen threshold |

## 11. The Research Journey: From Unsupervised to Supervised

### The Starting Point

Venator began with a strict unsupervised constraint: train only on benign prompts, never expose the detector to jailbreaks. The unsupervised ensemble (PCA + Mahalanobis, Isolation Forest, Autoencoder) achieved 0.70–0.87 AUROC depending on the method and attack category.

### What the Data Showed

The unsupervised ceiling is real. Unsupervised methods top out around 0.87 AUROC (autoencoder). The supervised linear probe blows past this to 0.999 AUROC with just 30 labeled examples. The gap is not close.

### The Honest Trade-Off

| | Unsupervised | Supervised (Linear Probe) |
|---|---|---|
| **Best AUROC** | 0.869 (Autoencoder) | 0.999 (Linear Probe) |
| **Training data** | Benign prompts only | Benign + 30 labeled jailbreaks |
| **Novel attack detection** | Catches anything unusual | Catches anything in the jailbreak direction |
| **Practical deployment** | No labels needed | Need ~30 labeled jailbreaks |
| **Label sensitivity** | None | Low (5 labels → 0.996 AUROC) |
| **Cross-source generalization** | N/A | Near-perfect (0.996+ AUROC) |

The supervised approach is strictly better when even a handful of labeled jailbreaks are available. The unsupervised detectors remain in the codebase for baseline comparison and for scenarios where no labeled jailbreaks exist.

## 12. Strengths and Limitations

### Strengths

- **Near-perfect detection** with minimal labeled data (0.999 AUROC with 30 examples)
- **Label-efficient**: 5 examples already achieve 0.996 AUROC
- **Fast inference**: scoring a single prompt takes ~1–3 seconds (dominated by MLX forward pass)
- **Interpretable**: the logistic regression weight vector defines the "jailbreak direction"
- **Cross-source generalization**: training on one attack type transfers to others
- **Runs locally**: entire pipeline on Apple Silicon via MLX, no cloud GPU needed

### Limitations

- **Requires access to model internals**: only works when you can extract hidden states (white-box setting)
- **Threshold sensitivity**: the operating point trades off FPR vs. recall
- **Distribution shift**: if the benign prompt distribution shifts significantly, the detector may need retraining
- **Apple Silicon dependency**: MLX is required for efficient local inference
- **MLP probe overfitting**: the more complex probe shows weaker generalization than logistic regression

## References

1. Mallen et al., "Eliciting Latent Knowledge from Quirky Language Models" (arXiv:2312.01037, 2024)
2. Burns et al., "Discovering Latent Knowledge Without Supervision" (arXiv:2212.03827, 2022)
3. Cunningham et al., "Cost-Effective Constitutional Classifiers via Representation Re-use" (Anthropic, 2025)
4. Sharma et al., "Constitutional Classifiers: Defending against Universal Jailbreak Attacks on LLMs" (arXiv:2501.18837, 2025)
5. Chao et al., "JailbreakBench: An Open Robustness Benchmark for Jailbreaking LLMs" (arXiv:2404.01318, 2024)
