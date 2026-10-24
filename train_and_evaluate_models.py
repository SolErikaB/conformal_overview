"""Train/evaluate every supported dataset, architecture, and imbalance mode.

Usage from the project directory:
    python train_and_evaluate_models.py --epochs 50

ImageNet models use torchvision's pretrained weights (the existing training
pipeline does not fine-tune ImageNet). Temperature variants are currently
implemented for ImageNet ResNet50 (T=0.5, 2, 4, and calibrated; T=1 is the plain outputs file).
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime

import torch
import yaml

import prepare_models

DATASETS = ("cifar10", "cifar100", "imagenet")
ARCHITECTURES = ("resnet18", "resnet50", "efficientnet_b0", "vit_b_16")
TEMPERATURE_TAGS = ("0_5", "2_0", "4_0", "calibrated")


def expected_evaluation_files(output_dir, dataset, architecture, is_imbalanced):
    suffix = "_imbalanced" if is_imbalanced else ""
    stem = f"{dataset}_{architecture}"
    paths = [
        os.path.join(output_dir, f"{stem}_outputs{suffix}.npz"),
        os.path.join(output_dir, f"{stem}_metrics{suffix}.npz"),
    ]
    if dataset == "imagenet" and architecture == "resnet50":
        paths.extend(
            os.path.join(output_dir, f"{stem}_{kind}_T_{tag}{suffix}.npz")
            for tag in TEMPERATURE_TAGS
            for kind in ("outputs", "metrics")
        )
    return paths


def write_training_summary(history, model_dir, dataset, architecture, is_imbalanced, seed, epochs):
    suffix = "_imbalanced" if is_imbalanced else ""
    dataset_dir = os.path.join(model_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    summary_path = os.path.join(dataset_dir, f"training_summary_{architecture}{suffix}.json")
    summary = {
        "dataset": dataset,
        "architecture": architecture,
        "simulate_class_imbalance": is_imbalanced,
        "epochs": epochs,
        "seed": seed,
        "train_loss": history["train_loss"],
        "train_accuracy": history["train_acc"],
        "validation_loss": history["val_loss"],
        "validation_accuracy": history["val_acc"],
        "test_loss": history["test_loss"],
        "test_accuracy": history["test_acc"],
        "config": history["config"],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    prepare_models.save_training_plots(
        history,
        architecture,
        dataset,
        output_path=dataset_dir,
        is_imbalanced=is_imbalanced,
    )
    return summary_path


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate all configured dataset/model/imbalance combinations sequentially."
    )
    parser.add_argument("--config", default="config.yaml", help="Base YAML config path.")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per trainable combination (default: 50).")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument(
        "--imbalance-modes",
        nargs="+",
        choices=("balanced", "imbalanced"),
        default=("balanced", "imbalanced"),
        help="Which evaluation/training modes to run (default: both).",
    )
    parser.add_argument("--force", action="store_true", help="Retrain and reevaluate even if artifacts exist.")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    training_cfg = base_config["training"]
    evaluation_cfg = base_config["evaluation"]
    data_dir = training_cfg["data_directory"]
    model_dir = training_cfg["model_directory"]
    output_dir = evaluation_cfg["output_directory"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    imbalance_modes = tuple(mode == "imbalanced" for mode in args.imbalance_modes)
    run_count = len(args.datasets) * len(args.architectures) * len(imbalance_modes)
    failures = []
    run_number = 0
    log_path = os.path.join(output_dir, f"run_{datetime.now():%Y%m%d_%H%M%S}.json")
    run_records = []

    for dataset in args.datasets:
        for architecture in args.architectures:
            for is_imbalanced in imbalance_modes:
                run_number += 1
                tag = "imbalanced" if is_imbalanced else "balanced"
                print(
                    f"\n[{run_number}/{run_count}] {dataset} / {architecture} / {tag}",
                    flush=True,
                )
                record = {
                    "dataset": dataset,
                    "architecture": architecture,
                    "simulate_class_imbalance": is_imbalanced,
                    "training": "not_applicable_pretrained_imagenet" if dataset == "imagenet" else "pending",
                    "evaluation": "pending",
                }

                suffix = "_imbalanced" if is_imbalanced else ""
                checkpoint_path = os.path.join(model_dir, dataset, f"{architecture}{suffix}.pth")
                try:
                    if dataset != "imagenet":
                        if args.force or not os.path.isfile(checkpoint_path):
                            prepare_models.set_seed(args.seed)
                            history = prepare_models.train_model(
                                arch_name=architecture,
                                dataset_name=dataset,
                                data_root=data_dir,
                                model_root=model_dir,
                                seed=args.seed,
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                learning_rate=args.learning_rate,
                                weight_decay=args.weight_decay,
                                num_workers=training_cfg.get("num_workers", 4),
                                simulate_class_imbalance=is_imbalanced,
                                minority_class_fraction=training_cfg.get("minority_class_fraction", 0.1),
                                minority_keep_fraction=training_cfg.get("minority_keep_fraction", 0.1),
                                minority_seed=training_cfg.get("imbalance_seed", args.seed),
                            )
                            record["training"] = "trained"
                            record["training_summary"] = write_training_summary(
                                history, model_dir, dataset, architecture,
                                is_imbalanced, args.seed, args.epochs,
                            )
                        else:
                            print(f"Checkpoint exists; skipping training: {checkpoint_path}", flush=True)
                            record["training"] = "skipped_checkpoint_exists"
                    else:
                        print(
                            "ImageNet training is not implemented by prepare_models.py; "
                            "evaluation will use torchvision pretrained weights.",
                            flush=True,
                        )

                    missing = [p for p in expected_evaluation_files(
                        output_dir, dataset, architecture, is_imbalanced
                    ) if not os.path.isfile(p)]
                    if args.force or missing:
                        run_config = copy.deepcopy(base_config)
                        run_config["training"]["dataset"] = dataset
                        run_config["training"]["model_architecture"] = architecture
                        run_config["training"]["simulate_class_imbalance"] = is_imbalanced
                        run_config["evaluation"]["dataset"] = dataset
                        run_config["evaluation"]["model_architecture"] = architecture
                        run_config["evaluation"]["simulate_class_imbalance"] = is_imbalanced

                        fd, temporary_config = tempfile.mkstemp(
                            prefix="run_config_", suffix=".yaml", dir=output_dir
                        )
                        os.close(fd)
                        try:
                            with open(temporary_config, "w") as f:
                                yaml.safe_dump(run_config, f, sort_keys=False)
                            subprocess.run(
                                [
                                    sys.executable,
                                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_eval_and_save_features.py"),
                                    "--config",
                                    temporary_config,
                                ],
                                cwd=os.path.dirname(os.path.abspath(__file__)),
                                check=True,
                            )
                        finally:
                            os.remove(temporary_config)
                        missing = [p for p in expected_evaluation_files(
                            output_dir, dataset, architecture, is_imbalanced
                        ) if not os.path.isfile(p)]
                        if missing:
                            raise RuntimeError("Evaluation did not create expected artifacts: " + ", ".join(missing))
                        record["evaluation"] = "evaluated"
                    else:
                        print("All expected evaluation artifacts exist; skipping evaluation.", flush=True)
                        record["evaluation"] = "skipped_artifacts_exist"

                except Exception as error:
                    record["error"] = repr(error)
                    failures.append(record)
                    print(f"FAILED: {error!r}", flush=True)
                finally:
                    run_records.append(record)
                    with open(log_path, "w") as f:
                        json.dump({"runs": run_records, "failures": failures}, f, indent=2)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    print(f"\nRun log: {log_path}")
    print(f"Completed {len(run_records) - len(failures)}/{len(run_records)} combinations; {len(failures)} failed.")
    if failures:
        for failure in failures:
            print(
                f"  {failure['dataset']} / {failure['architecture']} / "
                f"{'imbalanced' if failure['simulate_class_imbalance'] else 'balanced'}: {failure['error']}"
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
