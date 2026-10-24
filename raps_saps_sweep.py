"""Sweep RAPS and SAPS regularization parameters and save S-criterion results.

Sweep settings are hard-coded below; imbalance and temperature settings come from config.yaml.
APS-family scores are evaluated on probabilities with Euclidean distance.
The sweep runs once with fixed u=0.001 and once with randomized per-example
u=None; each mode has separate summary, detailed, and raw result files.
"""

import os
import time

import numpy as np
import pandas as pd
import yaml

from cp_eval import (
    ConformalConfig,
    breakpoint_values,
    compute_iteration,
    run_parallel_iterations,
    summarise,
)

CONFIG_PATH = "config.yaml"
N_ITERATIONS = 100
SWEEP_DATASETS = ("cifar10", "cifar100", "imagenet")
MODELS = ("resnet50",)
WINDOWS = (
    ("0", "e/4"),
    ("0", "e/2"),
    ("0", "e"),
    ("0", "1"),
    ("e/4", "e/2"),
    ("e/4", "e"),
    ("e/4", "1"),
    ("e/2", "e"),
    ("e/2", "1"),
    ("e", "1"),
)

# Parameter sweep grids for RAPS and SAPS.
RAPS_REG_K_GRID = [1, 2, 3, 4]
RAPS_REG_LAMBDA_GRID = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
SAPS_REG_LAMBDA_GRID = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]


def _canonicalized_value(value):
    if value is None or pd.isna(value):
        return "<missing>"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in {"true", "false"}:
            return value.lower()
        try:
            return format(float(value), ".15g")
        except ValueError:
            return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return format(float(value), ".15g")
    return str(value).strip()


def _same_identity_row_exists(table, identity):
    if table is None or table.empty:
        return False

    run_key = (
        "dataset",
        "model_architecture",
        "domain",
        "distance_metric",
        "mondrian",
        "simulate_class_imbalance",
        "minority_class_fraction",
        "calibration_minority_fraction",
        "imbalance_seed",
        "temperature",
        "u_mode",
        "u",
        "score_function",
        "reg_k",
        "reg_lambda",
    )
    if any(column not in table.columns or column not in identity for column in run_key):
        return False

    target = tuple(_canonicalized_value(identity[column]) for column in run_key)
    return any(
        tuple(_canonicalized_value(row[column]) for column in run_key) == target
        for _, row in table.iterrows()
    )


def _atomic_write_csv(table, path):
    temp_path = path + ".tmp"
    table.to_csv(temp_path, index=False)
    os.replace(temp_path, path)


def _write_result_tables(table, detailed_path, summary_path):
    _atomic_write_csv(table, detailed_path)
    summary = table.loc[table["window"] == "0-e"].copy()
    _atomic_write_csv(summary, summary_path)


def _safe_tag(value):
    return str(value).replace(".", "p").replace("-", "m").replace(" ", "")


def _parameter_plan():
    for reg_k in RAPS_REG_K_GRID:
        for reg_lambda in RAPS_REG_LAMBDA_GRID:
            yield "raps", reg_k, reg_lambda
    # SAPS has no k_reg parameter; do not duplicate its runs for different k.
    for reg_lambda in SAPS_REG_LAMBDA_GRID:
        yield "saps", np.nan, reg_lambda


def run_sweep():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    cp_cfg = config["conformal_prediction"]

    output_dir = os.path.join(config["evaluation"]["output_directory"], "figures")
    plan = list(_parameter_plan())
    os.makedirs(output_dir, exist_ok=True)
    failures = []
    total = len(SWEEP_DATASETS) * len(MODELS) * len(plan)

    for u_value, u_mode in ((0.001, "u_0p001"), (None, "u_random")):
        summary_csv_path = os.path.join(output_dir, f"RAPS_SAPS_sweep_{u_mode}.csv")
        detailed_csv_path = os.path.join(output_dir, f"RAPS_SAPS_sweep_{u_mode}_all_windows.csv")
        raw_dir = os.path.join(output_dir, "raps_saps_sweep_raw", u_mode)
        os.makedirs(raw_dir, exist_ok=True)
        if os.path.exists(detailed_csv_path):
            table = pd.read_csv(detailed_csv_path)
        else:
            table = None

        index = 0
        print(f"\nStarting RAPS/SAPS sweep with {u_mode} (u={u_value}).", flush=True)
        for dataset in SWEEP_DATASETS:
            for model in MODELS:
                try:
                    # The APS-family scores use probabilities and Euclidean distance.
                    conf = ConformalConfig(
                        CONFIG_PATH,
                        dataset=dataset,
                        model_architecture=model,
                        conformal_domain="probabilities",
                    )
                    conf.conformal_domain = "probabilities"
                    conf.distance_metric = "euclidean"
                    conf.u = u_value
                except Exception as error:
                    failures.append((u_mode, dataset, model, repr(error)))
                    print(f"Could not load {dataset}/{model}: {error!r}", flush=True)
                    continue

                temperature = cp_cfg.get("temperature", 1.0)
                context = {
                    "dataset": conf.dataset,
                    "model_architecture": conf.model_architecture,
                    "domain": "probabilities",
                    "distance_metric": "euclidean",
                    "mondrian": bool(conf.mondrian),
                    "simulate_class_imbalance": bool(conf.simulate_class_imbalance),
                    "minority_class_fraction": conf.minority_class_fraction,
                    "calibration_minority_fraction": conf.calibration_minority_fraction,
                    "imbalance_seed": conf.imbalance_seed,
                    "temperature": str(temperature),
                    "u_mode": u_mode,
                    "u": u_value,
                }
                print(
                    f"\nSweep target: dataset={dataset}, model={model}, "
                    f"imbalance={conf.simulate_class_imbalance}, u={u_mode}",
                    flush=True,
                )

                for score, reg_k, reg_lambda in plan:
                    index += 1
                    identity = {
                        **context,
                        "score_function": score,
                        "reg_k": reg_k,
                        "reg_lambda": reg_lambda,
                    }
                    if _same_identity_row_exists(table, identity):
                        print(
                            f"[{index}/{total}] Skipping already processed {u_mode} "
                            f"{dataset}/{model} {score}, k={reg_k}, lambda={reg_lambda}"
                        )
                        continue

                    conf.score_function = score
                    conf.reg_k = int(reg_k) if not pd.isna(reg_k) else 0
                    conf.reg_lambda = float(reg_lambda)
                    print(
                        f"[{index}/{total}] {u_mode} {dataset}/{model} {score}: "
                        f"reg_k={identity['reg_k']}, reg_lambda={reg_lambda}",
                        flush=True,
                    )
                    try:
                        computation_start = time.perf_counter()
                        results = run_parallel_iterations(
                            conf,
                            compute_iteration,
                            N_ITERATIONS,
                            desc=f"{u_mode} {dataset}/{model} {score} k={identity['reg_k']} lambda={reg_lambda}",
                        )
                        computation_time_seconds = time.perf_counter() - computation_start
                        s_matrix = np.asarray(
                            [[iteration[f"{lo}-{hi}"] for lo, hi in WINDOWS] for iteration in results],
                            dtype=float,
                        )

                        bp = breakpoint_values(conf.top1_accuracy)
                        widths = np.asarray([bp[hi] - bp[lo] for lo, hi in WINDOWS])
                        normalized_s = s_matrix / widths[None, :]
                        rows = [
                            {
                                **identity,
                                "reg_k": identity["reg_k"],
                                "window": f"{lo}-{hi}",
                                "lower": bp[lo],
                                "upper": bp[hi],
                                "n_iterations": N_ITERATIONS,
                                "computation_time_seconds": computation_time_seconds,
                                **summarise(normalized_s[:, window_index]),
                            }
                            for window_index, (lo, hi) in enumerate(WINDOWS)
                        ]
                        new_rows = pd.DataFrame(rows)
                        if table is None:
                            table = new_rows
                        else:
                            table = pd.concat([table, new_rows], ignore_index=True)
                        _write_result_tables(table, detailed_csv_path, summary_csv_path)
                        print("  Computation time: %.3f seconds" % computation_time_seconds, flush=True)

                        raw_tag = "_".join(
                            _safe_tag(identity[key])
                            for key in (
                                "dataset", "model_architecture", "domain", "score_function",
                                "reg_k", "reg_lambda", "simulate_class_imbalance",
                                "calibration_minority_fraction", "temperature", "u_mode",
                            )
                        )
                        raw_path = os.path.join(raw_dir, raw_tag + ".npz")
                        raw_metadata = dict(identity)
                        if raw_metadata["u"] is None:
                            raw_metadata["u"] = np.nan
                        np.savez_compressed(
                            raw_path,
                            s_matrix=s_matrix,
                            window_labels=np.asarray([f"{lo}-{hi}" for lo, hi in WINDOWS]),
                            iterations=np.arange(N_ITERATIONS),
                            **raw_metadata,
                        )
                    except Exception as error:
                        failures.append((u_mode, dataset, model, score, reg_k, reg_lambda, repr(error)))
                        print(
                            f"FAILED {u_mode} {dataset}/{model} {score}, "
                            f"k={reg_k}, lambda={reg_lambda}: {error!r}",
                            flush=True,
                        )

        print(f"{u_mode} summary (0-e only): {summary_csv_path}", flush=True)
        print(f"{u_mode} all-window results: {detailed_csv_path}", flush=True)
        print(f"{u_mode} raw per-iteration values: {raw_dir}", flush=True)
    if failures:
        print(f"Completed with {len(failures)} failures:", flush=True)
        for failure in failures:
            print("  " + " / ".join(map(str, failure)), flush=True)


if __name__ == "__main__":
    run_sweep()