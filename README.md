# Venator

**Jailbreak detection via a linear probe on LLM hidden state activations — 0.999 AUROC with only 30 labeled training examples.**

Venator trains a logistic regression on Mistral-7B's internal representations to detect jailbreak attempts, consistent with Anthropic's finding that linear probes on activations can match dedicated classifiers at a fraction of the cost ([Cunningham et al. 2025](https://www.anthropic.com)).

## Results

Primary detector: Linear probe on Mistral-7B layer 16 activations (50 PCA dims)

| Metric | Value |
|--------|-------|
| AUROC | 0.999 |
| AUPRC | 1.000 |
| Recall @ threshold | 0.99 |
| FPR @ threshold | 0.03 |
| Training jailbreaks needed | 30 |

A logistic regression probe achieves near-perfect jailbreak detection using the representations the LLM already computes. The ensemble of all detectors (0.932 AUROC) performs *worse* than the linear probe alone — ensembling adds noise when one component dominates.

## Core Idea

> Jailbreak attempts produce statistically distinguishable activation patterns in LLM hidden states — a logistic regression on middle-layer activations is sufficient to detect them.

This builds on findings from the ELK literature (Mallen et al. 2024, Burns et al. 2022) and Anthropic's safety research:

- Linear probes on middle-layer activations detect behavioral anomalies with 0.94+ AUROC
- Middle layers generalize best ("Earliest Informative Layer" criterion)
- Linear probes on activations match larger dedicated classifiers (Cunningham et al. 2025)

**Why this matters for AI safety:** Jailbreaks bypass safety training in unpredictable ways — we can't enumerate all attacks. A linear probe needs only ~30 labeled jailbreaks and reuses representations the LLM already computes. Probing internal activations captures *how the model computes*, not just *what it outputs*.

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │           Venator Pipeline               │
                        └─────────────────────────────────────────┘
                                         │
         ┌───────────────┬───────────────┼───────────────┬────────────────┐
         ▼               ▼               ▼               ▼                ▼
   ┌───────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐  ┌────────────┐
   │  Collect   │  │  Extract   │  │   Split    │  │   Train   │  │  Evaluate  │
   │  Prompts   │  │ Activations│  │  Data      │  │ Detectors │  │  & Detect  │
   └─────┬─────┘  └─────┬──────┘  └─────┬──────┘  └─────┬─────┘  └─────┬──────┘
         │               │               │               │              │
         ▼               ▼               ▼               ▼              ▼
   ┌───────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐  ┌────────────┐
   │  Benign   │  │   MLX +    │  │  Train:    │  │  Linear   │  │  AUROC,    │
   │  + Jail-  │  │ Mistral-7B │  │  benign +  │  │  Probe    │  │  PR curves │
   │  break    │  │ Hidden     │  │  ~30 jail- │  │ (primary) │  │  Prompt    │
   │  JSONL    │  │ States     │  │  breaks    │  │  + compar-│  │  Explorer  │
   │           │  │ → HDF5     │  │  Test:     │  │  ison     │  │  Figures   │
   │           │  │            │  │  held-out  │  │  baselines│  │            │
   └───────────┘  └────────────┘  └────────────┘  └───────────┘  └────────────┘
```

```
venator/
├── venator/
│   ├── config.py                 # Pydantic-settings configuration
│   ├── pipeline.py               # End-to-end orchestration
│   ├── activation/
│   │   ├── extractor.py          # MLX-based hidden state extraction
│   │   └── storage.py            # HDF5 activation storage
│   ├── data/
│   │   ├── prompts.py            # Prompt dataset management
│   │   └── splits.py             # Train/val/test splitting
│   ├── detection/
│   │   ├── base.py               # AnomalyDetector ABC
│   │   ├── linear_probe.py       # PRIMARY: logistic regression on activations
│   │   ├── contrastive.py        # Comparison: diff-in-means + class-conditional Mahalanobis
│   │   ├── pca_mahalanobis.py    # Comparison: unsupervised baseline
│   │   ├── isolation_forest.py   # Comparison: unsupervised baseline
│   │   ├── autoencoder.py        # Comparison: unsupervised baseline
│   │   ├── ensemble.py           # Training harness + comparison infrastructure
│   │   └── metrics.py            # AUROC, precision, recall, curves
│   └── dashboard/                # Streamlit 5-page pipeline UI
│       ├── app.py                # Main entry point + navigation
│       ├── state.py              # Session state management
│       ├── pages/                # Pipeline, Results, Explore, Detect, Ablations
│       └── components/           # Reusable charts, tables
├── scripts/                      # CLI entry points
│   ├── evaluate_final.py         # Definitive eval: JSON + figures + summary
│   └── ...                       # collect, extract, split, train, ablations
├── examples/
│   └── quickstart.py             # Minimal detection example
├── docs/                         # Methodology and results write-up
├── data/                         # Prompts and activation HDF5 files
├── models/                       # Saved detector models
└── tests/                        # 458 pytest tests
```

## Setup

**Requirements:** Python 3.11+, Apple Silicon Mac (for MLX acceleration)

```bash
# Clone and install
git clone <repo-url> && cd venator
pip install -e ".[dev,dashboard]"
```

**Dependencies:**

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| LLM Inference | MLX + mlx-lm | Optimized for Apple Silicon, native Metal acceleration |
| Model | Mistral-7B-Instruct-v0.3 (4-bit) | Fits M4 Pro memory, instruction-tuned |
| Numerical | NumPy, SciPy, scikit-learn | Standard scientific Python stack |
| Deep Learning | PyTorch (CPU) | Autoencoder detector only |
| Storage | HDF5 (h5py) | Efficient activation matrix storage |
| Dashboard | Streamlit + Plotly | Interactive pipeline UI |
| Figures | Matplotlib | Publication-ready static figures |
| Config | Pydantic Settings | Type-safe, environment variable overrides |

## Quickstart

### Option 1: Streamlit Dashboard (recommended)

```bash
streamlit run venator/dashboard/app.py
```

The dashboard guides you through a 5-page workflow:

1. **Pipeline** — Collect data, extract activations, create splits, train detectors
2. **Results** — Headline metrics, ROC curves, score distributions, detector comparison
3. **Explore** — Prompt-level deep dive, FP/FN error analysis
4. **Live Detection** — Single-prompt jailbreak detection with the primary detector
5. **Ablations** — Layer comparison, label efficiency, cross-source generalization

### Option 2: Python API

```python
from venator.pipeline import VenatorPipeline

# Load a trained pipeline and detect jailbreaks
pipeline = VenatorPipeline.load("models/v2/")
result = pipeline.detect("What causes the seasons to change?")
print(f"Score: {result['score']:.4f}, Jailbreak: {result['is_jailbreak']}")
```

See [examples/quickstart.py](examples/quickstart.py) for a complete example with multiple prompts.

### Option 3: CLI Scripts

```bash
# Train detector (requires extracted activations + splits)
python scripts/train_detector.py \
    --store data/activations/all.h5 \
    --splits data/splits_semi.json \
    --ensemble-type hybrid \
    --layer 16 \
    --output models/v2/

# Evaluate and generate publication figures
python scripts/evaluate_final.py \
    --model-dir models/v2/ \
    --store data/activations/all.h5 \
    --splits data/splits_semi.json \
    --output results/final/ \
    --figures results/figures/
```

<details>
<summary>Full CLI pipeline (all steps)</summary>

```bash
# 1. Collect prompts
python scripts/collect_prompts.py --output data/prompts/

# 2. Extract activations
python scripts/extract_activations.py \
    --prompts data/prompts/benign.jsonl \
    --output data/activations/all.h5 \
    --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
    --layers 12 14 16 18 20

# 3. Create splits (semi-supervised: small jailbreak set in train/val)
python scripts/create_splits.py \
    --store data/activations/all.h5 \
    --output data/splits_semi.json \
    --split-mode semi_supervised

# 4. Train
python scripts/train_detector.py \
    --store data/activations/all.h5 \
    --splits data/splits_semi.json \
    --ensemble-type hybrid \
    --layer 16 \
    --output models/v2/

# 5. Evaluate
python scripts/evaluate_final.py \
    --model-dir models/v2/ \
    --store data/activations/all.h5 \
    --splits data/splits_semi.json \
    --output results/final/ \
    --figures results/figures/

# 6. Run ablation studies
python scripts/run_supervised_ablations.py \
    --store data/activations/all.h5 \
    --splits data/splits_semi.json \
    --ablate label_budget generalization \
    --output results/supervised_ablations/
```

</details>

## Detection Architecture

The **linear probe** is the primary (production) detector. Other detectors exist for research comparison:

| Detector | Type | Role | AUROC |
|----------|------|------|-------|
| Linear Probe | Supervised | **Primary** | 0.999 |
| Contrastive Mahalanobis | Supervised | Comparison | 0.996 |
| Autoencoder | Unsupervised | Comparison (best unsupervised) | ~0.9 |
| PCA + Mahalanobis | Unsupervised | Comparison | ~0.9 |
| Isolation Forest | Unsupervised | Comparison | ~0.85 |
| Contrastive Direction | Supervised | Comparison | 0.656 |
| Ensemble (weighted avg) | Ensemble | Retired | 0.932 |

**Why no ensemble?** Ensembles help when component errors are decorrelated AND components are comparably accurate. When one component (the linear probe) is far stronger, ensembling adds noise. The linear probe at 0.999 AUROC is dragged down to 0.932 by including weaker detectors.

## Methodology

Semi-supervised with strict test set isolation — see [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the full write-up.

**Key principles:**

- **Train**: Benign prompts + ~30 labeled jailbreaks (semi-supervised)
- **Validation**: Benign + small jailbreak set — threshold via Youden's J statistic
- **Test**: Held-out benign + 70% of jailbreaks (never seen in training)
- **Primary metric**: AUROC (threshold-independent, standard for anomaly detection)

**Critical constraints:**
1. Majority of jailbreaks reserved for testing (70%+ never seen in training)
2. No threshold selection on test data
3. Healthy sample-to-feature ratio (PCA to 50 dims from 4096)
4. Primary detector only for production inference
5. Report AUROC as primary metric

## Configuration

All settings can be overridden via environment variables (prefix: `VENATOR_`):

| Setting | Default | Env Var |
|---------|---------|---------|
| Model | Mistral-7B-Instruct-v0.3 (4-bit) | `VENATOR_MODEL_ID` |
| Extraction layers | [12, 14, 16, 18, 20] | `VENATOR_EXTRACTION_LAYERS` |
| PCA dimensions | 50 | `VENATOR_PCA_DIMS` |
| Threshold percentile | 95.0 | `VENATOR_ANOMALY_THRESHOLD_PERCENTILE` |
| Random seed | 42 | `VENATOR_RANDOM_SEED` |

## Testing

```bash
# Run all tests (458 tests)
pytest tests/ -v

# With coverage
pytest tests/ --cov=venator

# Specific module
pytest tests/test_detectors.py -v
```

Test suite covers:
- Unit tests for each detector (including supervised types) on synthetic data
- Integration tests for the full pipeline with mock activations
- Methodology tests verifying no data leakage
- Roundtrip tests for save/load of all detectors
- Threshold calibration tests (Youden's J, F1, FPR target, percentile)

## References

- Mallen et al., ["Eliciting Latent Knowledge from Quirky Language Models"](https://arxiv.org/abs/2312.01037) (2024) — ELK paper; middle-layer probing
- Burns et al., ["Discovering Latent Knowledge Without Supervision"](https://arxiv.org/abs/2212.03827) (2022) — CCS; unsupervised truth discovery from activations
- Cunningham et al., "Cost-Effective Constitutional Classifiers via Representation Re-use" (Anthropic, 2025) — Linear probes as cheap monitors
- Sharma et al., ["Constitutional Classifiers: Defending against Universal Jailbreak Attacks on LLMs"](https://arxiv.org/abs/2501.18837) (2025) — Constitutional classifiers
- Chao et al., ["JailbreakBench: An Open Robustness Benchmark for Jailbreaking LLMs"](https://arxiv.org/abs/2404.01318) (2024) — Jailbreak dataset and evaluation framework

## License

MIT
