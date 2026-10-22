"""Measure ImageNet ResNet50 S-criteria across softmax temperatures.

The calibrated-temperature NPZ produced by model_eval_and_save_features.py
contains only CP examples; this script validates its saved source indices before
using it, so examples used to fit temperature cannot enter calibration or test.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import yaml

from cp_eval import (
    ConformalConfig,
    WINDOWS,
    WINDOW_LABELS,
    breakpoint_values,
    compute_iteration,
    run_parallel_iterations,
    summarise,
    summarise_coverage,
)


TEMPERATURES = (0.5, 1.0, 2.0, 4.0, "calibrated")
# These are the probability-domain score families used by cp_eval's score grid.
SCORE_FUNCTIONS = ("label", "mean", "margin", "aps", "raps", "saps")
# APS-family scores use no distance, so (as in cp_eval) they are run once with Euclidean.
DISTANCE_METRICS = ("euclidean", "cosine")
APS_FAMILY = ("aps", "raps", "saps")


def _canonical(value):
    if value is None or pd.isna(value):
        return "<missing>"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()
    if isinstance(value, (int, float, np.integer, np.floating)):
        return format(float(value), ".15g")
    text = str(value).strip()
    if text.lower() in ("true", "false"):
        return text.lower()
    try:
        return format(float(text), ".15g")
    except ValueError:
        return text


def _identity_mask(table, identity):
    if table is None or table.empty or any(key not in table.columns for key in identity):
        return np.zeros(0 if table is None else len(table), dtype=bool)
    mask = np.ones(len(table), dtype=bool)
    for key, value in identity.items():
        mask &= table[key].map(_canonical).to_numpy() == _canonical(value)
    return mask


def _is_complete(table, identity, n_iterations):
    if table is None or "window" not in table.columns or "n_iterations" not in table.columns:
        return False
    rows = table.loc[_identity_mask(table, identity)]
    return (
        set(rows["window"].astype(str)) == set(WINDOW_LABELS)
        and rows["n_iterations"].min() >= n_iterations
    )


def _atomic_write_csv(table, path):
    temporary_path = path + ".tmp"
    table.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def _write_tables(table, detailed_path, summary_path):
    _atomic_write_csv(table, detailed_path)
    _atomic_write_csv(table.loc[table["window"] == "0-e"].copy(), summary_path)


def _validate_calibrated_variant(conf):
    """Ensure temperature-fitting examples are absent from the CP source pool."""
    required = {"temperature_fit_indices", "cp_source_indices"}
    missing = required.difference(conf.data.files)
    if missing:
        raise ValueError(
            "The calibrated-temperature NPZ is missing provenance arrays "
            + ", ".join(sorted(missing))
            + ". Regenerate it with save_temperature_variants()."
        )

    fit_indices = np.asarray(conf.data["temperature_fit_indices"], dtype=np.int64)
    cp_indices = np.asarray(conf.data["cp_source_indices"], dtype=np.int64)
    if fit_indices.size == 0 or cp_indices.size != len(conf.data["labels"]):
        raise ValueError("Calibrated-temperature NPZ has inconsistent source-index metadata")
    if np.unique(fit_indices).size != fit_indices.size or np.unique(cp_indices).size != cp_indices.size:
        raise ValueError("Calibrated-temperature NPZ contains duplicate source indices")
    if np.intersect1d(fit_indices, cp_indices, assume_unique=True).size:
        raise ValueError(
            "Temperature-fitting examples overlap the calibrated CP data; refusing to run "
            "to avoid evaluation leakage."
        )
    return int(fit_indices.size), int(cp_indices.size)


def _configuration_identity(conf, temperature, score, reg_k, reg_lambda):
    return {
        "dataset": conf.dataset,
        "model_architecture": conf.model_architecture,
        "domain": "probabilities",
        "score_function": score,
        "distance_metric": conf.distance_metric,
        "mondrian": bool(conf.mondrian),
        "reg_k": reg_k,
        "reg_lambda": reg_lambda,
        "alpha": float(conf.alpha),
        "temperature": str(temperature),
        "u_mode": "u_random" if conf.u is None else "u_0p001" if np.isclose(float(conf.u), 0.001) else "u_" + str(conf.u),
        "u": np.nan if conf.u is None else float(conf.u),
        "simulate_class_imbalance": bool(conf.simulate_class_imbalance),
        "minority_class_fraction": conf.minority_class_fraction,
        "calibration_minority_fraction": conf.calibration_minority_fraction,
        "imbalance_seed": conf.imbalance_seed,
    }


def _score_distance_pairs():
    for score in SCORE_FUNCTIONS:
        for distance in (("euclidean",) if score in APS_FAMILY else DISTANCE_METRICS):
            yield score, distance


def _score_regularization(score, cp_cfg):
    if score == "raps":
        return int(cp_cfg["reg_k"]), float(cp_cfg["reg_lambda"])
    if score == "saps":
        return 0, float(cp_cfg["reg_lambda"])
    # APS is the unregularized RAPS special case; other scores ignore these.
    return 0, 0.0


def run_temperature_sweep(config_path="config.yaml", n_iterations=100):
    if n_iterations <= 0:
        raise ValueError("n_iterations must be positive")

    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    cp_cfg = config["conformal_prediction"]
    if cp_cfg.get("dataset") != "imagenet" or cp_cfg.get("model_architecture") != "resnet50":
        raise ValueError("The temperature sweep requires ImageNet ResNet50 evaluation artifacts")

    output_dir = os.path.join(config["evaluation"]["output_directory"], "figures")
    os.makedirs(output_dir, exist_ok=True)
    detailed_path = os.path.join(output_dir, "temperature_sweep_all_windows.csv")
    summary_path = os.path.join(output_dir, "temperature_sweep_0-e.csv")
    raw_dir = os.path.join(output_dir, "temperature_sweep_raw")
    os.makedirs(raw_dir, exist_ok=True)
    table = pd.read_csv(detailed_path) if os.path.exists(detailed_path) else None
    failures = []

    for temperature in TEMPERATURES:
        try:
            conf = ConformalConfig(
                config_path,
                dataset="imagenet",
                model_architecture="resnet50",
                conformal_domain="probabilities",
                temperature=temperature,
            )
            if temperature == "calibrated":
                fit_count, cp_source_count = _validate_calibrated_variant(conf)
            else:
                fit_count, cp_source_count = 0, len(conf.data["labels"])
        except Exception as error:
            failures.append((str(temperature), "load", repr(error)))
            print(f"Could not load T={temperature}: {error!r}", flush=True)
            continue

        print(
            f"\nTemperature T={temperature}: {len(conf.data['labels'])} CP-source examples, "
            f"n_calib={conf.n_calib}; excluded fit examples={fit_count}",
            flush=True,
        )
        for score, distance in _score_distance_pairs():
            reg_k, reg_lambda = _score_regularization(score, cp_cfg)
            conf.score_function = score
            conf.distance_metric = distance
            conf.reg_k = reg_k
            conf.reg_lambda = reg_lambda
            identity = _configuration_identity(conf, temperature, score, reg_k, reg_lambda)

            if _is_complete(table, identity, n_iterations):
                print(f"Skipping complete T={temperature}, score={score}, distance={distance}", flush=True)
                continue

            print(
                f"Running T={temperature}, score={score}, distance={distance} ({n_iterations} iterations)",
                flush=True,
            )
            try:
                start = time.perf_counter()
                results = run_parallel_iterations(
                    conf,
                    compute_iteration,
                    n_iterations,
                    desc=f"T={temperature} {score} {distance}",
                )
                elapsed = time.perf_counter() - start

                s_matrix = np.asarray(
                    [[result[label] for label in WINDOW_LABELS] for result in results],
                    dtype=float,
                )
                coverage = summarise_coverage([result["coverage"] for result in results], conf.alpha)
                breakpoints = breakpoint_values(conf.top1_accuracy)
                widths = np.asarray(
                    [breakpoints[upper] - breakpoints[lower] for lower, upper in WINDOWS]
                )
                normalized_s = s_matrix / widths[None, :]
                rows = [
                    {
                        **identity,
                        "reg_k": reg_k,
                        "reg_lambda": reg_lambda,
                        "top1_accuracy": conf.top1_accuracy,
                        "top5_accuracy": conf.top5_accuracy,
                        "n_calib": conf.n_calib,
                        "temperature_fit_examples_excluded": fit_count,
                        "cp_source_examples": cp_source_count,
                        "window": label,
                        "lower": breakpoints[lower],
                        "upper": breakpoints[upper],
                        "n_iterations": n_iterations,
                        "computation_time_seconds": elapsed,
                        **summarise(normalized_s[:, window_index]),
                        **coverage,
                    }
                    for window_index, ((lower, upper), label) in enumerate(zip(WINDOWS, WINDOW_LABELS))
                ]
                new_rows = pd.DataFrame(rows)
                if table is not None:
                    table = table.loc[~_identity_mask(table, identity)]
                    table = pd.concat([table, new_rows], ignore_index=True)
                else:
                    table = new_rows
                _write_tables(table, detailed_path, summary_path)

                raw_tag = "_".join(
                    str(identity[key]).replace(".", "p").replace("-", "m")
                    for key in ("temperature", "score_function", "distance_metric", "reg_k", "reg_lambda", "u_mode")
                )
                raw_path = os.path.join(raw_dir, raw_tag + ".npz")
                np.savez_compressed(
                    raw_path,
                    s_matrix=s_matrix,
                    window_labels=np.asarray(WINDOW_LABELS),
                    iterations=np.arange(n_iterations),
                    temperature=str(temperature),
                    temperature_fit_examples_excluded=fit_count,
                    cp_source_examples=cp_source_count,
                    **{key: value for key, value in identity.items() if key not in ("temperature",)},
                )
                print(f"  Median S for window 0-e: {rows[WINDOW_LABELS.index('0-e')]['S_median']:.4f}; "
                      f"elapsed={elapsed:.1f}s", flush=True)
            except Exception as error:
                failures.append((str(temperature), f"{score}/{distance}", repr(error)))
                print(f"FAILED T={temperature}, score={score}, distance={distance}: {error!r}", flush=True)

    print(f"Detailed results: {detailed_path}", flush=True)
    print(f"0-e summary: {summary_path}", flush=True)
    print(f"Raw per-iteration matrices: {raw_dir}", flush=True)
    if failures:
        print(f"Completed with {len(failures)} failures:", flush=True)
        for failure in failures:
            print("  " + " / ".join(failure), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Experiment YAML config")
    parser.add_argument("--iterations", type=int, default=100, help="Split redraws per configuration")
    args = parser.parse_args()
    run_temperature_sweep(args.config, args.iterations)