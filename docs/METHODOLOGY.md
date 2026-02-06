# Venator: Methodology

This document explains the research foundation, design decisions, and methodology constraints behind Venator's approach to jailbreak detection.

## 1. Background: The ELK Problem

**Eliciting Latent Knowledge (ELK)** asks: can we extract what an AI model "knows" from its internal representations, even when its outputs might be unreliable?

Mallen et al. (2024) showed that linear probes on middle-layer activations can detect when language models behave anomalously — with 0.94+ AUROC — by learning what "normal" computation looks like. Burns et al. (2022) demonstrated that unsupervised methods can discover latent knowledge without labeled data by exploiting the structure of internal representations.

Venator applies these findings to a concrete safety problem: detecting jailbreak attempts by treating them as anomalous activation patterns.

## 2. Why Unsupervised Anomaly Detection?

The standard approach to jailbreak detection is supervised classification: train a model to distinguish "jailbreak" from "not jailbreak" using labeled examples. This has fundamental limitations:

| Approach | Strengths | Weaknesses |
|----------|-----------|------------|
| **Supervised (classifier)** | High accuracy on known attacks | Fails on novel attacks; requires labeled jailbreaks for training |
| **Rule-based (keyword/pattern)** | Fast, explainable | Brittle; trivially evaded by paraphrasing |
| **Unsupervised (Venator)** | Detects novel attacks; no jailbreaks needed for training | Lower precision on edge cases; threshold sensitivity |

Unsupervised anomaly detection sidesteps the enumeration problem entirely. Instead of learning what jailbreaks look like (which changes constantly), we learn what *normal* prompts look like and flag anything sufficiently different. This catches novel attacks by construction, because any jailbreak — including ones we've never seen — must deviate from the learned normal distribution to affect model behavior.

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

Venator defaults to layers [12, 14, 16, 18, 20] for Mistral-7B (32 layers total). The ablation study on the dashboard's Ablations page empirically validates which layer performs best for a given dataset.

## 4. Why PCA Before Mahalanobis Distance?

Mistral-7B has a hidden dimension of 4096. With a typical training set of ~350 benign prompts (after splitting), the sample-to-feature ratio is approximately 350/4096 ≈ 0.085 — far below the minimum needed for reliable covariance estimation.

**The problem:** Mahalanobis distance requires inverting the covariance matrix. With more features than samples, the covariance matrix is singular (rank-deficient) and cannot be inverted. Even with regularization, the resulting distance metric is unreliable.

**The solution:** PCA reduces 4096 dimensions to 50, giving a sample-to-feature ratio of 350/50 = 7.0 — well above the recommended minimum of 5-10x. PCA also removes noise dimensions that don't contribute to the signal, focusing the Mahalanobis distance on the most informative variance directions.

```
Raw activations:     4096 dims, rank ≈ 350   → Covariance singular
After PCA (50 dims): 50 dims,   rank = 50    → Covariance well-conditioned
Sample/feature ratio: 350/50 = 7.0x          → Healthy for Mahalanobis
```

The optimal number of PCA components can be explored via the ablation studies. Too few components lose signal; too many reintroduce the sample-to-feature problem.

## 5. Ensemble Design: Decorrelated Errors

From the ELK paper: *"An ELK method can be useful even when it is no more accurate than other sources, as long as its errors are decorrelated."*

Venator combines three detectors chosen specifically for decorrelated error profiles:

### PCA + Mahalanobis Distance (weight: 2.0)

**Assumption:** Benign activations are approximately Gaussian in PCA space.

**Score:** Mahalanobis distance from the training mean, accounting for the covariance structure.

**Failure mode:** Misses anomalies that lie along high-variance directions of the benign distribution (i.e., prompts that are unusual but in a "Gaussian" way).

**Why primary (highest weight):** Validated by the ELK paper at 0.94+ AUROC. Theoretically principled for Gaussian-distributed data.

### Isolation Forest (weight: 1.5)

**Assumption:** Anomalies are easier to isolate (separate from the rest) via random axis-aligned splits.

**Score:** Average path length to isolate the point across many random trees (shorter path = more anomalous).

**Failure mode:** Struggles with clustered anomalies that form their own dense region. Axis-aligned splits miss diagonal anomaly boundaries.

**Why complementary:** No distributional assumption — captures anomalies that Mahalanobis misses because they violate the Gaussian assumption in specific ways.

### Autoencoder (weight: 1.0)

**Assumption:** Benign activations lie on a low-dimensional nonlinear manifold. The autoencoder learns this manifold; anomalies have high reconstruction error.

**Score:** Mean squared error between input and reconstructed activation in PCA space.

**Failure mode:** May memorize training data if overparameterized. Reconstruction error can be low for anomalies that happen to project well onto the learned manifold.

**Why complementary:** Captures nonlinear structure that PCA (linear) misses. Different training signal (reconstruction vs. distance vs. isolation).

### Score Normalization

Raw scores are on different scales:
- Mahalanobis distances: typically 0-100+
- Isolation Forest scores: typically -0.5 to 0.5
- Autoencoder MSE: typically 0-10

Before combining, each detector's scores are normalized to [0, 1] via **percentile rank** against the training set. This ensures equal-scale contribution regardless of raw score magnitude.

### Weighted Combination

```
ensemble_score = (2.0 × pca_maha_norm + 1.5 × iforest_norm + 1.0 × ae_norm) / 4.5
```

Weights reflect confidence in each detector based on the ELK literature and empirical performance. The ablation studies on the dashboard's Detector Comparison tab show whether the ensemble outperforms individual detectors on the user's specific data.

## 6. Strict Train/Test Methodology

This is the most important methodological constraint and the one most commonly violated in anomaly detection papers.

### The Rule

**Jailbreak prompts must NEVER appear in training or validation data.**

This is non-negotiable because:

1. **Anomaly detection assumes anomalies are absent from training.** If jailbreaks leak into training, the detector learns to accommodate them as "normal", reducing detection sensitivity.

2. **Threshold selection on test data is cheating.** The threshold must be set on validation data (which contains only benign prompts). Setting it on data that includes jailbreaks gives optimistically biased results.

3. **Information leakage invalidates AUROC.** If the detector has seen any jailbreaks during fitting (even indirectly through threshold tuning), the reported AUROC does not reflect real-world performance.

### Data Splits

```
                    ┌──────────────────────────────────────────────┐
                    │              Benign Prompts (N)               │
                    ├──────────────┬────────────┬──────────────────┤
                    │   Train      │    Val     │   Test (benign)  │
                    │   (70%)      │   (15%)    │     (15%)        │
                    │              │            │                  │
                    │  Fit PCA,    │  Set       │                  │
                    │  detectors   │  threshold │  Combined with   │
                    │              │            │  jailbreaks for  │
                    │              │            │  evaluation      │
                    └──────────────┴────────────┴──────────────────┘

                    ┌──────────────────────────────────────────────┐
                    │           Jailbreak Prompts (M)               │
                    ├──────────────────────────────────────────────┤
                    │              Test (jailbreak) — ALL           │
                    │                                              │
                    │  Never seen during training or validation    │
                    └──────────────────────────────────────────────┘
```

### Validation Checks

Venator's `SplitManager.validate_splits()` programmatically enforces:

- Training split contains zero jailbreak prompts
- Validation split contains zero jailbreak prompts
- All jailbreak prompts are in the test split
- No index overlap between any splits
- Sample-to-feature ratio is healthy (n_train / pca_dims >= 5)

The dashboard's Split page displays these checks as green/red indicators.

## 7. Evaluation Metrics

### Primary: AUROC

**Area Under the Receiver Operating Characteristic curve.** Measures the detector's ability to rank jailbreaks higher than benign prompts, across all possible thresholds.

- AUROC = 1.0: Perfect separation
- AUROC = 0.5: Random chance
- AUROC < 0.5: Worse than random (inverted scores)

AUROC is threshold-independent, making it the fairest comparison metric.

### Secondary Metrics

| Metric | What it measures | Reported at |
|--------|-----------------|-------------|
| AUPRC | Area Under Precision-Recall curve; better for imbalanced data | All thresholds |
| FPR@95TPR | False positive rate when 95% of jailbreaks are caught | Fixed TPR = 0.95 |
| Precision | Of flagged prompts, how many are actually jailbreaks? | Chosen threshold |
| Recall | Of all jailbreaks, how many were flagged? | Chosen threshold |
| F1 | Harmonic mean of precision and recall | Chosen threshold |

### Threshold Selection

The threshold is set at the **95th percentile of validation scores**. Since validation data is benign-only, this means:

- ~5% of benign prompts will be flagged (false positive rate ≈ 5%)
- The threshold is set entirely on "normal" data, with no information about jailbreaks

This percentile is configurable (90th-99th) in the dashboard's Train page.

## 8. Comparison to Supervised Approaches

| | Unsupervised (Venator) | Supervised (Probing Classifier) |
|---|---|---|
| **Training data** | Benign prompts only | Benign + jailbreak prompts |
| **Novel attack detection** | Yes — anything "unusual" is flagged | No — only attacks similar to training set |
| **Label requirement** | None for training | Requires labeled jailbreaks |
| **Typical AUROC** | 0.85-0.95 | 0.95-0.99 |
| **Failure mode** | False positives on unusual-but-benign prompts | Misses novel attack categories |
| **Maintenance** | No retraining needed for new attacks | Must retrain as attacks evolve |

Venator trades some accuracy for robustness to novel attacks. In practice, the unsupervised approach is most valuable as a **complementary signal** alongside supervised classifiers — its errors are decorrelated from supervised methods, improving overall system reliability.

## References

1. Mallen et al., "Eliciting Latent Knowledge from Quirky Language Models" (arXiv:2312.01037, 2024)
2. Burns et al., "Discovering Latent Knowledge Without Supervision" (arXiv:2212.03827, 2022)
3. Chao et al., "JailbreakBench: An Open Robustness Benchmark for Jailbreaking LLMs" (arXiv:2404.01318, 2024)
