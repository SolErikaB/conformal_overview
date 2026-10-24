#!/usr/bin/env python3
"""Run class-imbalance conformal-prediction evaluation."""

import argparse
import copy
import hashlib
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import conformal_prediction as cp_implementation
from cp_eval import ConformalConfig, ConformalPrediction

DATASETS = ("cifar10", "cifar100")
KEEP_FRACTION = 0.10
IMBALANCE_SEED = 123
CONTROL_TARGETS = {"cifar10": (10, 25, 50, 100, 500, 1000), "cifar100": (10, 25, 50, 100)}
RAPS_SAPS_SETTINGS = {
    "cifar10": {"raps": {"reg_k": 1, "reg_lambda": 0.002}, "saps": {"reg_lambda": 0.08}},
    "cifar100": {"raps": {"reg_k": 1, "reg_lambda": 0.01}, "saps": {"reg_lambda": 0.08}},
}
METHODS = (
    {"key": "label_prob_cosine", "score": "label", "domain": "probabilities", "metric": "cosine"},
    {"key": "margin_logits_euclidean", "score": "margin", "domain": "logits", "metric": "euclidean"},
    {"key": "margin_logits_cosine", "score": "margin", "domain": "logits", "metric": "cosine"},
    {"key": "mean_logits_euclidean", "score": "mean", "domain": "logits", "metric": "euclidean"},
    {"key": "mean_logits_cosine", "score": "mean", "domain": "logits", "metric": "cosine"},
    {"key": "mean_features_euclidean", "score": "mean", "domain": "features", "metric": "euclidean"},
    {"key": "mean_features_cosine", "score": "mean", "domain": "features", "metric": "cosine"},
    {"key": "aps_fixed", "score": "aps", "domain": "probabilities", "metric": None, "u": 0.001},
    {"key": "raps_fixed", "score": "raps", "domain": "probabilities", "metric": None, "u": 0.001},
    {"key": "saps_fixed", "score": "saps", "domain": "probabilities", "metric": None, "u": 0.001},
    {"key": "fast_gradient", "score": "fast_gradient", "domain": "features", "metric": "euclidean"},
)


def methods_for(dataset):
    settings = RAPS_SAPS_SETTINGS[dataset]
    return [
        {**method, **settings[method["score"]]} if method["score"] in settings else method
        for method in METHODS
    ]


def method_metadata(method):
    return {
        "method": method["key"],
        "score_function": method["score"],
        "domain": method["domain"],
        "distance_metric": method["metric"] or "not_applicable",
        "u": method.get("u"),
        "reg_k": method.get("reg_k"),
        "reg_lambda": method.get("reg_lambda"),
    }


def index_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def minimum_finite_class_calibration_size(alpha):
    n = 1
    while np.ceil((1.0 - alpha) * (n + 1)) > n:
        n += 1
    return n


def load_config(config_path, dataset):
    conf = ConformalConfig(
        config_path,
        dataset=dataset,
        model_architecture="resnet50",
        temperature=1.0,
        simulate_class_imbalance=True,
        imbalance_seed=IMBALANCE_SEED,
        calibration_minority_fraction=KEEP_FRACTION,
    )
    arrays = {name: np.asarray(conf.data[name]) for name in conf.data.files}
    for name in ("probabilities", "logits", "features"):
        arrays[name] = arrays[name].astype(np.float64)
    conf.data = arrays
    conf._arrays_cache = {}
    return conf


def create_split(conf, seed, domain, keep_fraction):
    conf.conformal_domain = domain
    conf.calibration_minority_fraction = keep_fraction
    return conf.create_split(seed, calibration_retention="floor_min_one")


def minority_class_ids(conf):
    arrays = conf._needed_arrays()
    classes = np.arange(arrays["probabilities"].shape[1])
    ids = ConformalConfig._minority_class_ids(
        arrays["labels"], conf.minority_class_fraction, conf.imbalance_seed, classes=classes
    )
    return np.asarray(sorted(ids), dtype=np.int64)


@contextmanager
def conformal_model_config(conf, method):
    """Expose the configured checkpoint to CP's existing model-backed score classes."""
    original_config = cp_implementation.config
    runtime_config = copy.deepcopy(conf.config)
    temporary_model_dir = None
    if method["score"] == "fast_gradient" and conf.simulate_class_imbalance:
        temporary_model_dir = tempfile.TemporaryDirectory()
        source = Path(runtime_config["training"]["model_directory"]) / conf.dataset
        source = source / f"{conf.model_architecture}_imbalanced.pth"
        if not source.is_file():
            temporary_model_dir.cleanup()
            raise FileNotFoundError(f"Missing imbalanced checkpoint for fast_gradient: {source}")
        destination = Path(temporary_model_dir.name) / conf.dataset
        destination.mkdir(parents=True)
        (destination / f"{conf.model_architecture}.pth").symlink_to(source.resolve())
        runtime_config["training"]["model_directory"] = temporary_model_dir.name

    cp_implementation.config = runtime_config
    try:
        yield
    finally:
        cp_implementation.config = original_config
        if temporary_model_dir is not None:
            temporary_model_dir.cleanup()


def create_conformal_predictor(conf, method, split, seed):
    with conformal_model_config(conf, method):
        cp = ConformalPrediction(
            alpha=1.0 - conf.top1_accuracy,
            calibration_data=split["calibration_data"],
            calibration_labels=split["calibration_labels"],
            calibration_preds=split["calibration_preds"],
            test_data=split["test_data"],
            test_labels=split["test_labels"],
            n_classes=conf.n_classes,
            distance_metric=method["metric"] or "euclidean",
            score_function=method["score"],
            mondrian=True,
            reg_k=method.get("reg_k", conf.reg_k),
            reg_lambda=method.get("reg_lambda", conf.reg_lambda),
            u=method.get("u", conf.u),
            random_state=seed,
            model_architecture=conf.model_architecture,
            dataset=conf.dataset,
        )
    cp.compute_scores()
    return cp


def prediction_mask(cp, alpha, mondrian):
    previous_mode = cp.mondrian
    try:
        cp.mondrian = mondrian
        thresholds = np.asarray(cp._thresholds_for_alpha(alpha), dtype=float)
        return cp._test_scores <= thresholds[None, :]
    finally:
        cp.mondrian = previous_mode


def summarize_metrics(cp, test_labels, minority_ids, alpha):
    labels = np.asarray(test_labels, dtype=np.int64)
    mondrian = prediction_mask(cp, alpha, mondrian=True)
    marginal = prediction_mask(cp, alpha, mondrian=False)
    minority = np.isin(labels, minority_ids)
    minority_rows = np.flatnonzero(minority)
    minority_labels = labels[minority]
    return {
        "minority_prevalence_mondrian": float(np.mean(mondrian[:, minority_ids].mean(axis=0))),
        "minority_prevalence_marginal": float(np.mean(marginal[:, minority_ids].mean(axis=0))),
        "minority_coverage_mondrian": float(np.mean(mondrian[minority_rows, minority_labels])),
        "minority_coverage_marginal": float(np.mean(marginal[minority_rows, minority_labels])),
        "minority_test_examples": int(minority.sum()),
    }


def run_main_dataset(conf, seeds):
    alpha = 1.0 - conf.top1_accuracy
    labels = np.asarray(conf.data["labels"])
    probabilities = np.asarray(conf.data["probabilities"])
    minority_ids = minority_class_ids(conf)
    minimum_n = minimum_finite_class_calibration_size(alpha)
    rows, count_records = [], []

    for seed in seeds:
        split = create_split(conf, seed, "probabilities", KEEP_FRACTION)
        calibration_idx, test_idx = split["calibration_idx"], split["test_idx"]
        calibration_counts = np.bincount(labels[calibration_idx], minlength=conf.n_classes)
        test_counts = np.bincount(labels[test_idx], minlength=conf.n_classes)
        minority_calibration_counts = {int(c): int(calibration_counts[c]) for c in minority_ids}
        minority_test_counts = {int(c): int(test_counts[c]) for c in minority_ids}
        hashes = {
            "calibration_pool_split_sha256": index_hash(split["calibration_pool_idx"]),
            "retained_calibration_split_sha256": index_hash(calibration_idx),
            "test_split_sha256": index_hash(test_idx),
        }
        minority_test = np.isin(labels[test_idx], minority_ids)
        predictions = probabilities[test_idx[minority_test]].argmax(axis=1)
        minority_accuracy = float(np.mean(predictions == labels[test_idx[minority_test]]))
        count_records.append({
            "dataset": conf.dataset,
            "seed": seed,
            "minority_class_calibration_counts": json.dumps(minority_calibration_counts, sort_keys=True),
            "minority_class_test_counts": json.dumps(minority_test_counts, sort_keys=True),
            "minority_calibration_n_min": min(minority_calibration_counts.values()),
            "minority_calibration_n_max": max(minority_calibration_counts.values()),
            "finite_threshold_min_n": minimum_n,
            "any_minority_n_below_cutoff": min(minority_calibration_counts.values()) < minimum_n,
            "minority_classes_below_cutoff": json.dumps(
                {c: n for c, n in minority_calibration_counts.items() if n < minimum_n}, sort_keys=True
            ),
            "minority_test_examples": sum(minority_test_counts.values()),
            **hashes,
        })

        for method in methods_for(conf.dataset):
            method_split = create_split(conf, seed, method["domain"], KEEP_FRACTION)
            cp = create_conformal_predictor(conf, method, method_split, seed)
            rows.append({
                "dataset": conf.dataset,
                "seed": seed,
                **method_metadata(method),
                "alpha": alpha,
                "model_top1_accuracy": conf.top1_accuracy,
                "minority_accuracy": minority_accuracy,
                "n_calib_pool": conf.n_calib,
                "n_calib_total": len(calibration_idx),
                "n_test_total": len(test_idx),
                **hashes,
                "minority_classes": json.dumps(list(map(int, minority_ids))),
                "minority_class_calibration_counts": count_records[-1]["minority_class_calibration_counts"],
                "minority_class_test_counts": count_records[-1]["minority_class_test_counts"],
                **summarize_metrics(cp, method_split["test_labels"], minority_ids, alpha),
                "expected_prevalence": float(np.mean(test_counts[minority_ids] / len(test_idx))),
            })
    return rows, count_records


def run_calibration_size_control(conf, seeds):
    alpha = 1.0 - conf.top1_accuracy
    labels = np.asarray(conf.data["labels"])
    minority_ids = minority_class_ids(conf)
    rows = []

    for seed in seeds:
        for target_n in CONTROL_TARGETS[conf.dataset]:
            fraction = min(1.0, target_n * conf.n_classes / conf.n_calib)
            for method in methods_for(conf.dataset):
                split = create_split(conf, seed, method["domain"], fraction)
                calibration_labels = split["calibration_labels"]
                class_counts = {int(c): int(np.sum(calibration_labels == c)) for c in minority_ids}
                cp = create_conformal_predictor(conf, method, split, seed)
                rows.append({
                    "dataset": conf.dataset,
                    "seed": seed,
                    **method_metadata(method),
                    "n_k": target_n,
                    "n_k_actual_median": float(np.median(list(class_counts.values()))),
                    "minority_class_calibration_counts": json.dumps(class_counts, sort_keys=True),
                    "alpha": alpha,
                    "model_top1_accuracy": conf.top1_accuracy,
                    "n_calib_majority": int(np.sum(~np.isin(calibration_labels, minority_ids))),
                    "n_test_total": len(split["test_idx"]),
                    "calibration_pool_split_sha256": index_hash(split["calibration_pool_idx"]),
                    "retained_calibration_split_sha256": index_hash(split["calibration_idx"]),
                    "test_split_sha256": index_hash(split["test_idx"]),
                    "minority_classes": json.dumps(list(map(int, minority_ids))),
                    **summarize_metrics(cp, split["test_labels"], minority_ids, alpha),
                })
    return rows


def run_class_imbalance_analysis(n_iterations=1000, config_path="config.yaml"):
    """Run class-imbalance calibration experiments and save per-seed output tables."""
    if n_iterations < 1:
        raise ValueError("n_iterations must be positive")
    with open(config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    output_dir = Path(config["evaluation"]["output_directory"]) / "figures" / "class_imbalance_review"
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = range(n_iterations)
    main_rows, calibration_records, control_rows = [], [], []

    for dataset in DATASETS:
        conf = load_config(config_path, dataset)
        print(f"Computing class-imbalance experiment for {dataset}...", flush=True)
        rows, counts = run_main_dataset(conf, seeds)
        main_rows.extend(rows)
        calibration_records.extend(counts)
        print(f"Computing calibration-size control for {dataset}...", flush=True)
        control_rows.extend(run_calibration_size_control(conf, seeds))

    main_table = pd.DataFrame(main_rows)
    control_table = pd.DataFrame(control_rows)
    for dataset, target_n in (("cifar10", 100), ("cifar100", 10)):
        main_part = main_table[main_table.dataset == dataset].set_index(["method", "seed"]).sort_index()
        control_part = control_table[
            (control_table.dataset == dataset) & (control_table.n_k == target_n)
        ].set_index(["method", "seed"]).sort_index()
        for metric in ("minority_prevalence_mondrian", "minority_coverage_mondrian"):
            if not np.allclose(main_part[metric], control_part[metric], atol=1e-12):
                raise AssertionError(f"Control {dataset} n_k={target_n} does not reproduce main result: {metric}")

    main_table.to_csv(output_dir / "main_per_run.csv", index=False)
    pd.DataFrame(calibration_records).to_csv(output_dir / "calibration_counts.csv", index=False)
    control_table.to_csv(output_dir / "control_per_run.csv", index=False)
    print(f"Wrote class-imbalance results to {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    run_class_imbalance_analysis(args.iterations, args.config)


if __name__ == "__main__":
    main()
