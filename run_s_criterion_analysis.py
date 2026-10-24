"""Run the configured grid of S-criterion evaluations."""

import argparse

import yaml

from cp_eval import (
    ConformalConfig,
    add_row_to_S_table,
    is_done,
    load_S_table,
)

DATASETS = ("cifar10", "cifar100", "imagenet")
MODELS = ("resnet50", "resnet18", "efficientnet_b0", "vit_b_16")
DOMAINS = ("probabilities", "logits", "features")
DISTANCES = ("euclidean", "cosine")
MONDRIAN_VALUES = (False,)

SCORE_SPECS = {
    "label": (("probabilities",), DISTANCES),
    "mean": (DOMAINS, DISTANCES),
    "margin": (("probabilities", "logits"), DISTANCES),
    "gradient": (("features",), ("euclidean",)),
    "fast_gradient": (("features",), ("euclidean",)),
    "aps": (("probabilities",), ("euclidean",)),
    "raps": (("probabilities",), ("euclidean",)),
    "saps": (("probabilities",), ("euclidean",)),
}

RAPS_SAPS_LAMBDAS = {
    "cifar10": {
        0.001: {"raps": 0.001, "saps": 0.008},
        None: {"raps": 0.1, "saps": 0.2},
    },
    "cifar100": {
        0.001: {"raps": 0.01, "saps": 0.08},
        None: {"raps": 0.1, "saps": 0.2},
    },
    "imagenet": {
        0.001: {"raps": 0.001, "saps": 0.04},
        None: {"raps": 0.01, "saps": 0.1},
    },
}
RAPS_SAPS_U_VALUES = (0.001, None)


def build_plan(reg_k, reg_lambda, default_u=0.001):
    """Build the active S-criterion configuration grid."""
    plan = []
    for dataset in DATASETS:
        for model in MODELS:
            for domain in DOMAINS:
                for score, (domains, distances) in SCORE_SPECS.items():
                    if domain not in domains:
                        continue
                    if score in ("raps", "saps"):
                        u_values = RAPS_SAPS_U_VALUES
                    elif score == "aps":
                        u_values = tuple(dict.fromkeys((default_u, None)))
                    else:
                        u_values = (default_u,)
                    for distance in distances:
                        for mondrian in MONDRIAN_VALUES:
                            for u in u_values:
                                regularization = reg_lambda
                                if score in ("raps", "saps"):
                                    try:
                                        regularization = RAPS_SAPS_LAMBDAS[dataset][u][score]
                                    except KeyError as error:
                                        raise ValueError(
                                            f"Missing RAPS/SAPS lambda for dataset={dataset}, U={u}, score={score}"
                                        ) from error
                                plan.append({
                                    "dataset": dataset,
                                    "model_architecture": model,
                                    "conformal_domain": domain,
                                    "score_function": score,
                                    "distance_metric": distance,
                                    "mondrian": mondrian,
                                    "reg_k": reg_k,
                                    "reg_lambda": regularization,
                                    "u": u,
                                })
    return plan


def run_S_criterion_analysis(n_iterations=100, config_path="config.yaml", dry_run=False):
    """Run or resume the S-criterion grid and update the configured output directory."""
    if n_iterations < 1:
        raise ValueError("n_iterations must be positive")
    with open(config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    cp_config = config["conformal_prediction"]
    table_path = f"{config['evaluation']['output_directory']}/figures/S_table.csv"
    plan = build_plan(cp_config["reg_k"], cp_config["reg_lambda"], cp_config.get("u", 0.001))
    table = load_S_table(table_path)
    pending = [spec for spec in plan if not is_done(table, spec, n_iterations)]

    print(f"{len(plan)} configurations planned, {len(plan) - len(pending)} already complete, {len(pending)} pending")
    if dry_run:
        for spec in pending:
            print(f"  todo: {spec}")
        return

    failures = []
    conf = None
    current_group = None
    completed = 0
    for index, spec in enumerate(pending, 1):
        group = (spec["dataset"], spec["model_architecture"], spec["conformal_domain"])
        if group != current_group:
            conf = None
            current_group = group
            try:
                conf = ConformalConfig(
                    config_path,
                    dataset=spec["dataset"],
                    model_architecture=spec["model_architecture"],
                    conformal_domain=spec["conformal_domain"],
                    temperature=1.0,
                    simulate_class_imbalance=False,
                )
            except Exception as error:
                print(f"FAILED to load {group}: {error!r}", flush=True)

        if conf is None:
            failures.append((str(spec), "could not load data"))
            continue

        conf.score_function = spec["score_function"]
        conf.distance_metric = spec["distance_metric"]
        conf.mondrian = spec["mondrian"]
        conf.reg_k = spec["reg_k"]
        conf.reg_lambda = spec["reg_lambda"]
        conf.u = spec["u"]
        print(f"[{index}/{len(pending)}] {spec}", flush=True)
        try:
            add_row_to_S_table(n_iterations, conf)
            completed += 1
        except Exception as error:
            print(f"FAILED: {spec}: {error!r}", flush=True)
            failures.append((str(spec), repr(error)))

    print(f"Finished: {completed} computed, {len(failures)} failed", flush=True)
    for description, reason in failures:
        print(f"  failed: {description} -> {reason}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_S_criterion_analysis(args.iterations, args.config, args.dry_run)


if __name__ == "__main__":
    main()
