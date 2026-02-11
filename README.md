# Venator

**Jailbreak detection through activation probing.**

Venator is an automated pipeline for training and optimising detectors that identify jailbreak attempts on language models by reading their hidden activations during inference. Rather than analysing the text of a prompt or trusting the model's output, it looks at what the language model is doing internally. It appears jailbreaks leave a very clear signature.

A linear probe (logistic regression) on middle-layer activations from Mistral-7B achieves **0.999 AUROC** with around 30 labelled jailbreak examples. Even using just five labelled jailbreak examples achieves an AUROC of 0.996.

## How it works

During inference, Venator extracts hidden state activations from the transformer's middle layers — the point where the model has moved past token-level processing and started representing input intent. These 4096-dimensional vectors get reduced to 50 via PCA (you can go as low as 20 without meaningful loss), then scored by the probe. The entire scoring step is a single matrix multiply.

The pipeline runs locally on Apple Silicon via MLX with 4-bit quantised models. No cloud GPUs needed. Though one of the next questions to be answered is whether the smaller models can be used to flag jailbreaks for larger models.

## Results

**Model:** Mistral-7B-Instruct-v0.3 (4-bit quantised via MLX) | **Layer:** 18 | **PCA:** 50 dims | **Test set:** 150 benign + 350 jailbreak prompts (held-out)

| Metric | Linear Probe |
|--------|-------------|
| AUROC | 0.999 |
| AUPRC | 1.000 |
| F1 | 0.991 |
| Precision | 0.997 |
| Recall | 0.986 |
| FPR | 0.007 |

The best unsupervised method (autoencoder) reached 0.869 AUROC. The supervised probe with just 5 labelled examples already sits at 0.996.

### Full detector comparison

| Detector | Type | AUROC | F1 |
|----------|------|-------|-----|
| **Linear Probe** | **Supervised** | **0.999** | **0.991** |
| MLP Probe | Supervised | 0.997 | 0.990 |
| Contrastive Mahalanobis | Supervised | 0.995 | 0.975 |
| Custom Ensemble | Ensemble | 0.967 | 0.970 |
| Autoencoder | Unsupervised | 0.869 | 0.878 |
| PCA + Mahalanobis | Unsupervised | 0.737 | 0.823 |
| Isolation Forest | Unsupervised | 0.711 | 0.787 |
| Contrastive Direction | Supervised | 0.656 | 0.880 |

<details>
<summary>Score distributions and ROC curves</summary>

**Linear probe score distribution** — benign prompts cluster near 0, jailbreaks near 1, with clean separation at the 0.550 threshold:

![Linear Probe Scores](venator/results/linearprobescores.png)

**ROC curve comparison** — all 8 detectors overlaid:

![ROC Curve Comparison](venator/results/ROC%20curve%20comparison.png)

**Precision-recall curve** (linear probe):

![Precision-Recall Curve](venator/results/precision-recallcurve.png)

</details>

## What I found interesting

**The jailbreak boundary is linear.** A Multi-Layer Perceptron (MLP) probe with considerably more capacity scored 0.997 — no improvement over logistic regression. The separation in activation space is a straight line.

**The signal appears early.** Layer 4 out of 32 gives 0.981 AUROC. By layer 10 it's at 0.998+. The model recognises jailbreak intent almost immediately.

**It generalises across attack types.** Training on one jailbreak category and testing on entirely different ones (i.e. DAN-style or Encoding jailbreaks): 0.996–0.999 AUROC. There seems to be a indentifiable "jailbreak direction" in activation space that's consistent regardless of technique.

**Ensembling reduced performance.** Combining all detectors dropped performance from 0.999 to 0.967. The weaker methods just added noise in the cases I tested.

**Jailbreak signal has low dimensionality.** The probe is remarkably insensitive to how aggressively you compress the activations. Raw 4096-dimensional activations score 0.9995 AUROC. PCA down to 50 dimensions: 0.9995. Down to 20: 0.9991. Even 10 dimensions still gives 0.996. The jailbreak signal is low-dimensional — most of those 4096 features are noise as far as detection is concerned.

**Labelled training data is very efficient.** Five labelled jailbreaks give 0.994 AUROC. Thirty gets you to 0.997. Seventy-five reaches 0.9995. For comparison, the best unsupervised methods without any labels top out at 0.812 (autoencoder) and 0.695 (PCA + Mahalanobis). The curve is nearly flat from 5 to 30 labels — the probe identifies the jailbreak direction from the first few examples and additional labels offer diminishing returns.

| Labelled Jailbreaks | Linear Probe AUROC |
|---------------------|-------------------|
| 5 | 0.996 |
| 10 | 0.996 |
| 20 | 0.997 |
| 30 | 0.998 |
| 50 | 0.999 |

## Background

This project was influenced by Anthropic's research direction on applying anomaly detection to model latent activations to flag out-of-distribution inputs like jailbreaks. The approach builds on the Cheap Monitors paper (Cunningham et al., 2025), which showed that reusing a model's own intermediate representations for safety classification can match dedicated classifiers at a fraction of the cost.

It started as a purely unsupervised system, to see if labelled data could be eschewed. But adding even a handful of labelled examples made the unsupervised approach redundant, so the focus shifted to understanding how few labels you actually need and how well the probe generalises.

Full write-ups: [Methodology](docs/METHODOLOGY.md) | [Results](docs/RESULTS.md)

## Getting started

**Requirements:** Python 3.11+, Apple Silicon Mac (for MLX acceleration)

```bash
git clone https://github.com/peterstringer/venator.git && cd venator
pip install -e ".[dev,dashboard]"
```

### Dashboard (recommended)

```bash
streamlit run venator/dashboard/app.py
```

The **Quick Run** page (default landing) automates the full pipeline: data collection, activation extraction, splitting, auto-optimisation across layers/PCA/detectors, and final evaluation. Other pages provide manual pipeline control, results exploration, prompt-level analysis, live detection, and ablation studies.

### Python API

```python
from venator.pipeline import VenatorPipeline

pipeline = VenatorPipeline.load("models/v2/")
result = pipeline.detect("What causes the seasons to change?")
print(f"Score: {result['score']:.4f}, Jailbreak: {result['is_jailbreak']}")
```

### CLI

<details>
<summary>Full CLI pipeline</summary>

```bash
# 1. Collect prompts
python scripts/collect_prompts.py --output data/prompts/

# 2. Extract activations
python scripts/extract_activations.py \
    --prompts data/prompts/benign.jsonl \
    --output data/activations/all.h5 \
    --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
    --layers 4 6 8 10 12 14 16 18 20 22 24

# 3. Create splits
python scripts/create_splits.py \
    --store data/activations/all.h5 \
    --output data/splits.json

# 4. Train
python scripts/train_detector.py \
    --store data/activations/all.h5 \
    --splits data/splits.json \
    --ensemble-type hybrid \
    --layer 18 \
    --output models/v2/

# 5. Evaluate
python scripts/evaluate_final.py \
    --model-dir models/v2/ \
    --store data/activations/all.h5 \
    --splits data/splits.json \
    --output results/final/

# 6. Ablation studies
python scripts/run_supervised_ablations.py \
    --store data/activations/all.h5 \
    --splits data/splits.json \
    --ablate label_budget generalization \
    --output results/supervised_ablations/
```

</details>

## Tech

| Component | Technology |
|-----------|-----------|
| LLM inference | MLX + mlx-lm (Apple Silicon, Metal acceleration) |
| Model | Mistral-7B-Instruct-v0.3, 4-bit quantised |
| ML | scikit-learn (PCA, logistic regression), PyTorch (MLP probe, autoencoder) |
| Storage | HDF5 via h5py |
| Dashboard | Streamlit + Plotly |
| Config | Pydantic Settings (`VENATOR_` env prefix) |

## References

- Cunningham et al., "Cost-Effective Constitutional Classifiers via Representation Re-use" (Anthropic, 2025) — Linear probes as cheap monitors
- Mallen et al., ["Eliciting Latent Knowledge from Quirky Language Models"](https://arxiv.org/abs/2312.01037) (2024) — Middle-layer probing for ELK
- Burns et al., ["Discovering Latent Knowledge Without Supervision"](https://arxiv.org/abs/2212.03827) (2022) — Unsupervised truth discovery from activations
- Sharma et al., ["Constitutional Classifiers"](https://arxiv.org/abs/2501.18837) (2025) — Defending against universal jailbreak attacks
- Chao et al., ["JailbreakBench"](https://arxiv.org/abs/2404.01318) (2024) — Jailbreak dataset and evaluation framework

## Licence

MIT
