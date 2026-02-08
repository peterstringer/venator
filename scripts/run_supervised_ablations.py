#!/usr/bin/env python3
"""Supervised ablation studies: labeled data budget, ensemble comparison, and generalization.

Extends the unsupervised ablations (run_ablations.py) with experiments that
require labeled jailbreak data. Key questions answered:

1. **Label budget**: How few labeled jailbreaks do supervised detectors need?
2. **Ensemble comparison**: Unsupervised vs supervised vs hybrid on same data.
3. **Layer sensitivity**: Does the optimal layer change with labeled data?
4. **Generalization**: Do supervised detectors generalize to unseen attack types?

Requires semi-supervised splits (from create_splits.py --split-mode semi_supervised).

Usage:
    python scripts/run_supervised_ablations.py \
        --store data/activations/all.h5 \
        --splits data/splits_semi.json \
        --output results/supervised_ablations/

    python scripts/run_supervised_ablations.py \
        --store data/activations/all.h5 \
        --splits data/splits_semi.json \
        --ablate label_budget ensemble_comparison layers \
        --output results/supervised_ablations/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path so we can import venator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from venator.activation.storage import ActivationStore
from venator.config import VenatorConfig
from venator.data.splits import SplitManager
from venator.detection.autoencoder import AutoencoderDetector
from venator.detection.contrastive import (
    ContrastiveDirectionDetector,
    ContrastiveMahalanobisDetector,
)
from venator.detection.ensemble import (
    DetectorEnsemble,
    DetectorType,
    create_default_ensemble,
    create_hybrid_ensemble,
    create_supervised_ensemble,
    create_unsupervised_ensemble,
)
from venator.detection.isolation_forest import IsolationForestDetector
from venator.detection.linear_probe import LinearProbeDetector, MLPProbeDetector
from venator.detection.metrics import evaluate_detector
from venator.detection.pca_mahalanobis import PCAMahalanobisDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Configurations
ABLATION_LAYERS = [8, 10, 12, 14, 16, 18, 20, 22, 24]
LABEL_BUDGETS = [3, 5, 10, 20, 30, 50, 100]


def _get_test_data(
    store: ActivationStore,
    splits: dict,
    layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Get stacked test data and labels for a given layer."""
    X_benign = store.get_activations(
        layer, indices=splits["test_benign"].indices.tolist()
    )
    X_jailbreak = store.get_activations(
        layer, indices=splits["test_jailbreak"].indices.tolist()
    )
    X_test = np.vstack([X_benign, X_jailbreak])
    labels = np.concatenate([
        np.zeros(len(X_benign), dtype=np.int64),
        np.ones(len(X_jailbreak), dtype=np.int64),
    ])
    return X_test, labels


# -----------------------------------------------------------------------
# Ablation 1: Label budget — how few labeled jailbreaks do we need?
# -----------------------------------------------------------------------


def ablate_label_budget(
    store: ActivationStore,
    splits: dict,
    config: VenatorConfig,
    layer: int,
    seed: int = 42,
) -> list[dict]:
    """Test supervised detectors with varying numbers of labeled jailbreaks.

    Key question: "How few labeled examples do we need?"
    Trains each supervised detector with 3, 5, 10, 20, 30, 50, 100 labeled
    jailbreaks and reports AUROC at each budget level.
    """
    X_train_benign = store.get_activations(
        layer, indices=splits["train_benign"].indices.tolist()
    )
    X_val_benign = store.get_activations(
        layer, indices=splits["val_benign"].indices.tolist()
    )
    X_train_jailbreak_full = store.get_activations(
        layer, indices=splits["train_jailbreak"].indices.tolist()
    )
    X_val_jailbreak = store.get_activations(
        layer, indices=splits["val_jailbreak"].indices.tolist()
    )
    X_test, test_labels = _get_test_data(store, splits, layer)

    n_available = len(X_train_jailbreak_full)
    rng = np.random.default_rng(seed)

    # Detectors to test at each budget level
    detector_configs = [
        ("linear_probe", lambda: LinearProbeDetector()),
        ("contrastive_dir", lambda: ContrastiveDirectionDetector()),
        ("contrastive_mahal", lambda: ContrastiveMahalanobisDetector()),
    ]

    results = []
    for n_labeled in LABEL_BUDGETS:
        if n_labeled > n_available:
            logger.info(
                "Skipping budget=%d (only %d jailbreaks available)",
                n_labeled, n_available,
            )
            continue

        # Subsample jailbreak training data
        indices = rng.choice(n_available, size=n_labeled, replace=False)
        X_train_jailbreak = X_train_jailbreak_full[indices]

        for det_name, det_factory in detector_configs:
            logger.info("Label budget: %d jailbreaks, detector=%s...", n_labeled, det_name)
            t0 = time.perf_counter()

            ensemble = DetectorEnsemble(
                threshold_percentile=config.anomaly_threshold_percentile,
            )
            ensemble.add_detector(
                det_name, det_factory(), weight=1.0,
                detector_type=DetectorType.SUPERVISED,
            )

            try:
                ensemble.fit(
                    X_train_benign, X_val_benign,
                    X_train_jailbreak=X_train_jailbreak,
                    X_val_jailbreak=X_val_jailbreak,
                )

                result = ensemble.score(X_test)
                metrics = evaluate_detector(
                    result.ensemble_scores, test_labels,
                    threshold=ensemble.threshold_,
                )
                elapsed = time.perf_counter() - t0

                results.append({
                    "n_labeled": n_labeled,
                    "detector": det_name,
                    "auroc": metrics["auroc"],
                    "auprc": metrics["auprc"],
                    "f1": metrics.get("f1", 0.0),
                    "time_s": round(elapsed, 2),
                })
            except Exception as e:
                logger.warning(
                    "Failed: budget=%d, detector=%s: %s", n_labeled, det_name, e,
                )
                results.append({
                    "n_labeled": n_labeled,
                    "detector": det_name,
                    "auroc": 0.0,
                    "auprc": 0.0,
                    "f1": 0.0,
                    "error": str(e),
                })

    return results


# -----------------------------------------------------------------------
# Ablation 2: Ensemble comparison — unsupervised vs supervised vs hybrid
# -----------------------------------------------------------------------


def ablate_ensemble_comparison(
    store: ActivationStore,
    splits: dict,
    config: VenatorConfig,
    layer: int,
) -> list[dict]:
    """Compare unsupervised, supervised, and hybrid ensembles on same data.

    Runs all three ensemble types and reports side-by-side metrics.
    """
    X_train_benign = store.get_activations(
        layer, indices=splits["train_benign"].indices.tolist()
    )
    X_val_benign = store.get_activations(
        layer, indices=splits["val_benign"].indices.tolist()
    )
    X_train_jailbreak = store.get_activations(
        layer, indices=splits["train_jailbreak"].indices.tolist()
    )
    X_val_jailbreak = store.get_activations(
        layer, indices=splits["val_jailbreak"].indices.tolist()
    )
    X_test, test_labels = _get_test_data(store, splits, layer)

    ensemble_configs = [
        ("unsupervised", create_unsupervised_ensemble, False),
        ("supervised", create_supervised_ensemble, True),
        ("hybrid", create_hybrid_ensemble, True),
    ]

    results = []
    for name, factory, needs_jailbreak in ensemble_configs:
        logger.info("Ensemble comparison: %s...", name)
        t0 = time.perf_counter()

        ensemble = factory(threshold_percentile=config.anomaly_threshold_percentile)
        ensemble.fit(
            X_train_benign, X_val_benign,
            X_train_jailbreak=X_train_jailbreak if needs_jailbreak else None,
            X_val_jailbreak=X_val_jailbreak if needs_jailbreak else None,
        )

        result = ensemble.score(X_test)
        metrics = evaluate_detector(
            result.ensemble_scores, test_labels, threshold=ensemble.threshold_,
        )
        elapsed = time.perf_counter() - t0

        # Also compute per-detector AUROC
        per_det = {}
        for det_result in result.detector_results:
            det_metrics = evaluate_detector(det_result.normalized_scores, test_labels)
            per_det[det_result.name] = det_metrics["auroc"]

        results.append({
            "ensemble": name,
            "n_detectors": len(ensemble.detectors),
            "detectors": ", ".join(n for n, _, _ in ensemble.detectors),
            "auroc": metrics["auroc"],
            "auprc": metrics["auprc"],
            "f1": metrics.get("f1", 0.0),
            "fpr_at_95_tpr": metrics["fpr_at_95_tpr"],
            "per_detector_auroc": per_det,
            "time_s": round(elapsed, 2),
        })

    return results


# -----------------------------------------------------------------------
# Ablation 3: Layer comparison for supervised detectors
# -----------------------------------------------------------------------


def ablate_supervised_layers(
    store: ActivationStore,
    splits: dict,
    config: VenatorConfig,
) -> list[dict]:
    """Test if the optimal layer changes when using labeled data.

    Compares PCA+Mahalanobis (unsupervised) vs ContrastiveDirection (supervised)
    at each layer to see if supervised detectors prefer different layers.
    """
    available_layers = set(store.layers)
    test_layers = [layer for layer in ABLATION_LAYERS if layer in available_layers]

    if not test_layers:
        logger.warning("No ablation layers found in store (available: %s)", sorted(available_layers))
        return []

    results = []
    for layer in test_layers:
        logger.info("Supervised layer ablation: layer %d...", layer)

        X_train_benign = store.get_activations(
            layer, indices=splits["train_benign"].indices.tolist()
        )
        X_val_benign = store.get_activations(
            layer, indices=splits["val_benign"].indices.tolist()
        )
        X_train_jailbreak = store.get_activations(
            layer, indices=splits["train_jailbreak"].indices.tolist()
        )
        X_val_jailbreak = store.get_activations(
            layer, indices=splits["val_jailbreak"].indices.tolist()
        )
        X_test, test_labels = _get_test_data(store, splits, layer)

        # Unsupervised baseline: PCA + Mahalanobis
        e_unsup = DetectorEnsemble(
            threshold_percentile=config.anomaly_threshold_percentile,
        )
        e_unsup.add_detector(
            "pca_mahalanobis",
            PCAMahalanobisDetector(n_components=config.pca_dims),
        )
        e_unsup.fit(X_train_benign, X_val_benign)
        unsup_result = e_unsup.score(X_test)
        unsup_metrics = evaluate_detector(unsup_result.ensemble_scores, test_labels)

        # Supervised: ContrastiveDirection
        e_sup = DetectorEnsemble(
            threshold_percentile=config.anomaly_threshold_percentile,
        )
        e_sup.add_detector(
            "contrastive_direction",
            ContrastiveDirectionDetector(),
            detector_type=DetectorType.SUPERVISED,
        )
        e_sup.fit(
            X_train_benign, X_val_benign,
            X_train_jailbreak=X_train_jailbreak,
            X_val_jailbreak=X_val_jailbreak,
        )
        sup_result = e_sup.score(X_test)
        sup_metrics = evaluate_detector(sup_result.ensemble_scores, test_labels)

        results.append({
            "layer": layer,
            "unsupervised_auroc": unsup_metrics["auroc"],
            "supervised_auroc": sup_metrics["auroc"],
            "delta": sup_metrics["auroc"] - unsup_metrics["auroc"],
        })

    return results


# -----------------------------------------------------------------------
# Ablation 4: Generalization to unseen jailbreak types
# -----------------------------------------------------------------------


def ablate_generalization(
    store: ActivationStore,
    splits: dict,
    config: VenatorConfig,
    layer: int,
) -> list[dict]:
    """Test if supervised detectors generalize to unseen jailbreak types.

    If jailbreak prompts are labeled with source metadata, this would split
    by source and train on some, test on others. Since we may not have source
    labels, this falls back to a leave-K-out approach: train on a random subset
    of jailbreaks, test on the held-out subset. If the detector learns
    "jailbreak-ness" in general (not just specific patterns), AUROC on
    held-out types should remain high.

    This tests the ELK paper's hypothesis that activation-level anomalies
    are attack-type agnostic.
    """
    X_train_benign = store.get_activations(
        layer, indices=splits["train_benign"].indices.tolist()
    )
    X_val_benign = store.get_activations(
        layer, indices=splits["val_benign"].indices.tolist()
    )
    X_test_benign = store.get_activations(
        layer, indices=splits["test_benign"].indices.tolist()
    )

    # Combine all available jailbreak data for cross-validation
    all_jailbreak_indices = np.concatenate([
        splits["train_jailbreak"].indices,
        splits["val_jailbreak"].indices,
        splits["test_jailbreak"].indices,
    ])
    X_all_jailbreak = store.get_activations(layer, indices=all_jailbreak_indices.tolist())

    n_jailbreak = len(X_all_jailbreak)
    if n_jailbreak < 10:
        logger.warning(
            "Too few jailbreaks (%d) for generalization ablation", n_jailbreak,
        )
        return []

    rng = np.random.default_rng(config.random_seed)
    perm = rng.permutation(n_jailbreak)

    # K-fold: split jailbreaks into 3 groups, train on 2, test on 1
    n_folds = 3
    fold_size = n_jailbreak // n_folds

    results = []
    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n_jailbreak
        test_idx = perm[test_start:test_end]
        train_idx = np.concatenate([perm[:test_start], perm[test_end:]])

        # Further split train_idx into train (80%) and val (20%)
        n_train_jb = max(3, int(len(train_idx) * 0.8))
        train_jb_idx = train_idx[:n_train_jb]
        val_jb_idx = train_idx[n_train_jb:]

        if len(val_jb_idx) < 1:
            val_jb_idx = train_jb_idx[:1]

        X_train_jailbreak = X_all_jailbreak[train_jb_idx]
        X_val_jailbreak = X_all_jailbreak[val_jb_idx]
        X_test_jailbreak = X_all_jailbreak[test_idx]

        # Build test set: held-out benign + held-out jailbreaks
        X_test = np.vstack([X_test_benign, X_test_jailbreak])
        test_labels = np.concatenate([
            np.zeros(len(X_test_benign), dtype=np.int64),
            np.ones(len(X_test_jailbreak), dtype=np.int64),
        ])

        logger.info(
            "Generalization fold %d: train=%d jailbreaks, test=%d jailbreaks",
            fold, len(train_jb_idx), len(test_idx),
        )

        # Train contrastive detector on known jailbreaks
        e_sup = DetectorEnsemble(
            threshold_percentile=config.anomaly_threshold_percentile,
        )
        e_sup.add_detector(
            "contrastive_direction",
            ContrastiveDirectionDetector(),
            detector_type=DetectorType.SUPERVISED,
        )
        e_sup.fit(
            X_train_benign, X_val_benign,
            X_train_jailbreak=X_train_jailbreak,
            X_val_jailbreak=X_val_jailbreak,
        )

        sup_result = e_sup.score(X_test)
        sup_metrics = evaluate_detector(
            sup_result.ensemble_scores, test_labels,
            threshold=e_sup.threshold_,
        )

        # Unsupervised baseline for comparison (doesn't use jailbreak data)
        e_unsup = DetectorEnsemble(
            threshold_percentile=config.anomaly_threshold_percentile,
        )
        e_unsup.add_detector(
            "pca_mahalanobis",
            PCAMahalanobisDetector(n_components=config.pca_dims),
        )
        e_unsup.fit(X_train_benign, X_val_benign)

        unsup_result = e_unsup.score(X_test)
        unsup_metrics = evaluate_detector(
            unsup_result.ensemble_scores, test_labels,
            threshold=e_unsup.threshold_,
        )

        results.append({
            "fold": fold,
            "n_train_jailbreaks": len(train_jb_idx),
            "n_test_jailbreaks": len(test_idx),
            "supervised_auroc": sup_metrics["auroc"],
            "supervised_f1": sup_metrics.get("f1", 0.0),
            "unsupervised_auroc": unsup_metrics["auroc"],
            "unsupervised_f1": unsup_metrics.get("f1", 0.0),
            "delta_auroc": sup_metrics["auroc"] - unsup_metrics["auroc"],
        })

    return results


# -----------------------------------------------------------------------
# Output formatting
# -----------------------------------------------------------------------


def _print_table(title: str, results: list[dict], columns: list[tuple[str, str, int]]) -> None:
    """Print a formatted comparison table.

    Args:
        title: Table title.
        results: List of row dicts.
        columns: List of (key, header, width) tuples.
    """
    if not results:
        print(f"\n{title}: No results")
        return

    print(f"\n{'=' * 75}")
    print(title)
    print("=" * 75)

    header = "  " + " | ".join(f"{h:>{w}}" for _, h, w in columns)
    print(header)
    print("  " + "-" * (sum(w + 3 for _, _, w in columns) - 1))

    for row in results:
        cells = []
        for key, _, w in columns:
            val = row.get(key, "")
            if isinstance(val, float):
                cells.append(f"{val:>{w}.4f}")
            else:
                cells.append(f"{str(val):>{w}}")
        print("  " + " | ".join(cells))

    print("=" * 75)


def main() -> None:
    config = VenatorConfig()
    default_layer = config.extraction_layers[len(config.extraction_layers) // 2]

    parser = argparse.ArgumentParser(
        description="Run supervised ablation studies"
    )
    parser.add_argument(
        "--store",
        type=Path,
        required=True,
        help="Path to the HDF5 activation store",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        required=True,
        help="Path to semi-supervised split definitions (JSON)",
    )
    parser.add_argument(
        "--ablate",
        nargs="+",
        choices=["label_budget", "ensemble_comparison", "layers", "generalization"],
        default=["label_budget", "ensemble_comparison", "layers", "generalization"],
        help="Which ablations to run (default: all)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=default_layer,
        help=f"Base layer for budget/comparison ablations (default: {default_layer})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for JSON results",
    )
    args = parser.parse_args()

    if not args.store.exists():
        logger.error("Store not found: %s", args.store)
        sys.exit(1)
    if not args.splits.exists():
        logger.error("Splits file not found: %s", args.splits)
        sys.exit(1)

    store = ActivationStore(args.store)
    splits = SplitManager.load_splits(args.splits)

    # Validate we have semi-supervised splits
    if "train_benign" not in splits:
        logger.error(
            "This script requires semi-supervised splits "
            "(with train_benign/train_jailbreak keys). "
            "Use create_splits.py --split-mode semi_supervised first."
        )
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Store: %s", store)
    logger.info("Ablations to run: %s", args.ablate)
    logger.info(
        "Training jailbreaks available: %d",
        splits["train_jailbreak"].n_samples,
    )

    all_results = {}

    # --- 1. Label budget ablation ---
    if "label_budget" in args.ablate:
        logger.info("Running label budget ablation...")
        budget_results = ablate_label_budget(
            store, splits, config, args.layer,
        )
        all_results["label_budget"] = budget_results

        with open(output_dir / "ablation_label_budget.json", "w", encoding="utf-8") as f:
            json.dump(budget_results, f, indent=2)

        _print_table(
            "Label Budget Ablation (AUROC vs N labeled jailbreaks)",
            budget_results,
            [
                ("n_labeled", "N_labels", 8),
                ("detector", "Detector", 20),
                ("auroc", "AUROC", 7),
                ("auprc", "AUPRC", 7),
                ("time_s", "Time(s)", 7),
            ],
        )

    # --- 2. Ensemble comparison ---
    if "ensemble_comparison" in args.ablate:
        logger.info("Running ensemble comparison on layer %d...", args.layer)
        comparison_results = ablate_ensemble_comparison(
            store, splits, config, args.layer,
        )
        all_results["ensemble_comparison"] = comparison_results

        with open(output_dir / "ablation_ensemble_comparison.json", "w", encoding="utf-8") as f:
            json.dump(comparison_results, f, indent=2)

        _print_table(
            f"Ensemble Comparison (layer={args.layer})",
            comparison_results,
            [
                ("ensemble", "Type", 15),
                ("auroc", "AUROC", 7),
                ("auprc", "AUPRC", 7),
                ("f1", "F1", 7),
                ("fpr_at_95_tpr", "FPR@95", 7),
                ("time_s", "Time(s)", 7),
            ],
        )

    # --- 3. Layer comparison for supervised detectors ---
    if "layers" in args.ablate:
        logger.info("Running supervised layer ablation...")
        layer_results = ablate_supervised_layers(store, splits, config)
        all_results["supervised_layers"] = layer_results

        with open(output_dir / "ablation_supervised_layers.json", "w", encoding="utf-8") as f:
            json.dump(layer_results, f, indent=2)

        _print_table(
            "Layer Ablation: Unsupervised vs Supervised",
            layer_results,
            [
                ("layer", "Layer", 6),
                ("unsupervised_auroc", "Unsup AUROC", 11),
                ("supervised_auroc", "Sup AUROC", 11),
                ("delta", "Delta", 7),
            ],
        )

        # Report optimal layers
        if layer_results:
            best_unsup = max(layer_results, key=lambda r: r["unsupervised_auroc"])
            best_sup = max(layer_results, key=lambda r: r["supervised_auroc"])
            print(f"\n  Best unsupervised layer: {best_unsup['layer']} "
                  f"(AUROC={best_unsup['unsupervised_auroc']:.4f})")
            print(f"  Best supervised layer:   {best_sup['layer']} "
                  f"(AUROC={best_sup['supervised_auroc']:.4f})")
            if best_unsup["layer"] != best_sup["layer"]:
                print("  Note: optimal layer differs — supervised detectors "
                      "may benefit from different layer selection.")
            else:
                print("  Optimal layer is the same for both — consistent "
                      "with ELK 'earliest informative layer' finding.")

    # --- 4. Generalization to unseen jailbreak types ---
    if "generalization" in args.ablate:
        logger.info("Running generalization ablation...")
        gen_results = ablate_generalization(
            store, splits, config, args.layer,
        )
        all_results["generalization"] = gen_results

        with open(output_dir / "ablation_generalization.json", "w", encoding="utf-8") as f:
            json.dump(gen_results, f, indent=2)

        _print_table(
            "Generalization: Train on subset, test on held-out jailbreaks",
            gen_results,
            [
                ("fold", "Fold", 5),
                ("n_train_jailbreaks", "Train JB", 9),
                ("n_test_jailbreaks", "Test JB", 8),
                ("supervised_auroc", "Sup AUROC", 11),
                ("unsupervised_auroc", "Unsup AUROC", 11),
                ("delta_auroc", "Delta", 7),
            ],
        )

        # Summary
        if gen_results:
            mean_sup = np.mean([r["supervised_auroc"] for r in gen_results])
            mean_unsup = np.mean([r["unsupervised_auroc"] for r in gen_results])
            print(f"\n  Mean supervised AUROC:    {mean_sup:.4f}")
            print(f"  Mean unsupervised AUROC:  {mean_unsup:.4f}")
            if mean_sup > mean_unsup:
                print("  Supervised detectors generalize to held-out jailbreaks — "
                      "consistent with learning 'jailbreak-ness' rather than "
                      "memorizing specific attack patterns.")
            else:
                print("  Unsupervised detectors perform comparably — "
                      "labeled data may not help for generalization on this dataset.")

    # Save combined results
    with open(output_dir / "ablation_supervised_all.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
