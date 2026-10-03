import prepare_models
import yaml
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


class FeatureExtractorWrapper(nn.Module):
    """Wrapper that extracts features from last two layers along with logits"""
    
    def __init__(self, model, arch_name):
        super().__init__()
        self.arch_name = arch_name
        
        # Handle ModelWithUpsample wrapper
        if hasattr(model, 'base_model'):
            self.upsample = model.upsample
            base_model = model.base_model
            self.has_upsample = True
        else:
            self.upsample = None
            base_model = model
            self.has_upsample = False
        
        # Split model into features and classifier based on architecture
        if 'resnet' in arch_name:
            # ResNet structure: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
            children = list(base_model.children())
            
            # Features up to layer3 (second-to-last feature layer)
            self.features = nn.Sequential(*children[:-1])
            self.classifier = children[-1]  # fc
            
        elif 'efficientnet' in arch_name:
            # EfficientNet structure: features (Sequential), avgpool, classifier
            children = list(base_model.children())

            # Features up to and including avgpool
            self.features = nn.Sequential(*children[:-1])
            self.classifier = children[-1]  # classifier
            
        elif 'vit' in arch_name:
            class ViTFeatureExtractor(nn.Module):
                """Reproduces vit_b_16's forward() up to (and including) the CLS token,
                i.e. everything before the classification head."""
                def __init__(self, vit_model):
                    super().__init__()
                    self.vit = vit_model

                def forward(self, x):
                    x = self.vit._process_input(x)
                    n = x.shape[0]
                    batch_class_token = self.vit.class_token.expand(n, -1, -1)
                    x = torch.cat([batch_class_token, x], dim=1)
                    x = self.vit.encoder(x)
                    x = x[:, 0]  # CLS token -> [batch, 768]
                    return x

            self.features = ViTFeatureExtractor(base_model)
            self.classifier = base_model.heads

        else:
            raise ValueError(f"Unsupported architecture: {arch_name}")
    
    def forward(self, x, return_features=False):
        # Apply upsample if needed
        if self.has_upsample:
            x = self.upsample(x)

        # Extract features
        features = self.features(x)
        features = torch.flatten(features, 1)
            
        # Get logits
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set and extract all outputs.
    
    Returns:
        dict with keys: 'logits', 'probabilities', 'features',
                       'labels', 'predictions'
    """
    model.eval()
    
    all_logits = []
    all_probs = []
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            
            # Get logits and features from both layers
            logits, features = model(images, return_features=True)
            
            # Compute probabilities (softmax of logits)
            probs = torch.softmax(logits, dim=1)
            
            # Store results
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Concatenate all batches
    results = {
        'logits': np.concatenate(all_logits, axis=0),
        'probabilities': np.concatenate(all_probs, axis=0),
        'features': np.concatenate(all_features, axis=0),
        'labels': np.concatenate(all_labels, axis=0),
    }
    
    # Add predictions
    results['predictions'] = np.argmax(results['logits'], axis=1)
    
    return results


def save_results(results, save_dir, dataset, model_architecture, is_imbalanced=False):
    """Save evaluation results to disk"""
    os.makedirs(save_dir, exist_ok=True)

    suffix = '_imbalanced' if is_imbalanced else ''
    # Save main results
    save_path = os.path.join(save_dir, f"{dataset}_{model_architecture}_outputs{suffix}.npz")
    np.savez_compressed(
        save_path,
        logits=results['logits'],
        probabilities=results['probabilities'],
        features=results['features'],
        labels=results['labels'],
        predictions=results['predictions']
    )
    
    print(f"\nSaved outputs to: {save_path}")
    print(f"  - Logits shape: {results['logits'].shape}")
    print(f"  - Probabilities shape: {results['probabilities'].shape}")
    print(f"  - Features shape: {results['features'].shape}")
    print(f"  - Labels shape: {results['labels'].shape}")
    
    return save_path


def fit_temperature(logits, labels):
    """Fit a scalar temperature by minimizing multiclass negative log-likelihood."""
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp

    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    rows = np.arange(len(labels))

    def mean_nll(log_temperature):
        scaled_logits = logits / np.exp(log_temperature)
        return np.mean(logsumexp(scaled_logits, axis=1) - scaled_logits[rows, labels])

    result = minimize_scalar(
        mean_nll,
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
        options={"xatol": 1e-5},
    )
    if not result.success:
        raise RuntimeError("Temperature fitting failed: " + result.message)
    return float(np.exp(result.x))


def save_temperature_variants(results, save_dir, dataset, model_architecture,
                              temperatures=(0.5, 1.0, 2.0, 4.0),
                              fit_holdout_size=10000, random_seed=42,
                              is_imbalanced=False):
    """Save per-temperature NPZs for ImageNet ResNet50.

    Fixed-T files include all evaluation examples. The calibrated file fits T
    on a deterministic 10k holdout and contains only the remaining CP examples.
    """
    if dataset != "imagenet" or model_architecture != "resnet50":
        raise ValueError("Temperature variants are supported only for ImageNet ResNet50")

    os.makedirs(save_dir, exist_ok=True)
    logits = np.asarray(results["logits"])
    labels = np.asarray(results["labels"])
    features = np.asarray(results["features"])
    predictions = np.asarray(results["predictions"])
    n_samples = len(labels)

    def save_variant(temperature, indices, tag, fit_indices=None):
        selected_logits = logits[indices]
        scaled_logits = selected_logits.astype(np.float64) / temperature
        scaled_logits -= scaled_logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(scaled_logits)
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        imbalance_suffix = '_imbalanced' if is_imbalanced else ''
        stem = "%s_%s" % (dataset, model_architecture)
        outputs_path = os.path.join(save_dir, f"{stem}_outputs_T_{tag}{imbalance_suffix}.npz")
        payload = {
            "logits": scaled_logits,
            "raw_logits": selected_logits,
            "probabilities": probabilities,
            "features": features[indices],
            "labels": labels[indices],
            "predictions": predictions[indices],
            "temperature": np.asarray(temperature),
        }
        if fit_indices is not None:
            payload["temperature_fit_indices"] = fit_indices
            payload["cp_source_indices"] = indices
        np.savez_compressed(outputs_path, **payload)

        metrics_path = os.path.join(save_dir, f"{stem}_metrics_T_{tag}{imbalance_suffix}.npz")
        np.savez_compressed(metrics_path, **compute_metrics({
            "logits": selected_logits,
            "predictions": predictions[indices],
            "labels": labels[indices],
            "features": features[indices],
        }))
        print("Saved T=%s outputs and metrics to %s" % (tag, outputs_path), flush=True)

    all_indices = np.arange(n_samples)
    for temperature in temperatures:
        temperature = float(temperature)
        if temperature <= 0:
            raise ValueError("Temperatures must be positive")
        tag = str(temperature).replace(".", "_")
        save_variant(temperature, all_indices, tag)

    if n_samples <= fit_holdout_size:
        raise ValueError(
            "Need more than %d examples to fit a temperature and retain CP data; found %d"
            % (fit_holdout_size, n_samples)
        )
    order = np.random.RandomState(random_seed).permutation(n_samples)
    fit_indices = np.sort(order[:fit_holdout_size])
    cp_indices = np.sort(order[fit_holdout_size:])
    calibrated_temperature = fit_temperature(logits[fit_indices], labels[fit_indices])
    print("Fitted calibrated T=%.6f on %d examples; excluded them from the %d CP rows."
          % (calibrated_temperature, len(fit_indices), len(cp_indices)), flush=True)
    save_variant(calibrated_temperature, cp_indices, "calibrated", fit_indices=fit_indices)


def compute_metrics(results):
    """Compute and display evaluation metrics"""
    predictions = results['predictions']
    labels = results['labels']
    
    # Top 1 and top 5 accuracy
    top_1_accuracy = (predictions == labels).mean()
    top_5_accuracy = np.mean([
        labels[i] in np.argsort(results['logits'][i])[-5:] for i in range(len(labels))
    ])
    
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"Top 1 Accuracy: {top_1_accuracy:.6f} ({top_1_accuracy*100:.4f}%)")
    print(f"Top 5 Accuracy: {top_5_accuracy:.6f} ({top_5_accuracy*100:.4f}%)")
    print(f"Features shape: {results['features'].shape}")
    print("="*50)
    
    return {
        'top_1_accuracy': top_1_accuracy,
        'top_5_accuracy': top_5_accuracy,
    }


if __name__ == "__main__":
    
    seed = 123
    prepare_models.set_seed(seed)
    
    parser = argparse.ArgumentParser(description="Evaluate a trained model and save features/logits.")
    parser.add_argument("--config", default="config.yaml", help="Path to the experiment YAML config.")
    args = parser.parse_args()

    # Load config file with system-specific parameters
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset = config['evaluation']['dataset']
    model_architecture = config['evaluation']['model_architecture']
    
    data_dir = config['training']['data_directory']
    model_dir = config['training']['model_directory']
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test dataset
    print(f"\nLoading dataset: {dataset}")
    is_imbalanced = bool(config.get('evaluation', {}).get(
        'simulate_class_imbalance',
        config.get('training', {}).get('simulate_class_imbalance', False),
    ))
    simulate_imbalance = is_imbalanced
    minority_class_fraction = config.get('training', {}).get('minority_class_fraction', 0.1)
    minority_keep_fraction = config.get('training', {}).get('minority_keep_fraction', 0.1)
    minority_seed = config.get('training', {}).get('imbalance_seed', seed)
    _, _, test_set, num_classes, input_size = prepare_models.get_datasets(
        dataset,
        data_dir,
        seed,
        simulate_imbalance=simulate_imbalance,
        keep_fraction=minority_keep_fraction,
        minority_class_fraction=minority_class_fraction,
        minority_seed=minority_seed,
    )
    print(f"Test set size: {len(test_set)}")
    print(f"Number of classes: {num_classes}")
    print(f"Input size: {input_size}")
   
    
    # Create test dataloader
    test_loader = DataLoader(
        test_set,
        batch_size=16,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    
    # Load model
    print(f"\nLoading model: {model_architecture}")
    model = prepare_models.get_model(model_architecture, dataset, num_classes, input_size)
    
    if dataset != 'imagenet':
        imbalance_suffix = '_imbalanced' if is_imbalanced else ''
        model_path = os.path.join(
            model_dir, dataset, f"{model_architecture}{imbalance_suffix}.pth"
        )
        opposite_suffix = '' if is_imbalanced else '_imbalanced'
        opposite_model_path = os.path.join(
            model_dir, dataset, f"{model_architecture}{opposite_suffix}.pth"
        )
        if not os.path.exists(model_path) and os.path.exists(opposite_model_path):
            raise FileNotFoundError(
                f"Evaluation is configured for {'imbalanced' if is_imbalanced else 'balanced'} data, "
                f"so it expects checkpoint {model_path!r}; only the opposite-mode checkpoint exists: "
                f"{opposite_model_path!r}. Check evaluation.simulate_class_imbalance and the checkpoint name."
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}. "
                "Check evaluation.simulate_class_imbalance and train/save the matching model first."
            )
        print(f"Model path: {model_path}")
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Wrap model for feature extraction
    print("\nWrapping model for feature extraction...")
    model = FeatureExtractorWrapper(model, model_architecture)
    model = model.to(device)
    model.eval()
    
    # Evaluate model
    print("\nStarting evaluation...")
    results = evaluate_model(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    # Save results
    output_dir = config.get('evaluation', {}).get('output_directory', './evaluation_outputs')
    save_path = save_results(results, output_dir, dataset, model_architecture, is_imbalanced=is_imbalanced)

    if dataset == 'imagenet' and model_architecture == 'resnet50':
        save_temperature_variants(results, output_dir, dataset, model_architecture, is_imbalanced=is_imbalanced)

    # Optional: Save metrics separately
    suffix = '_imbalanced' if is_imbalanced else ''
    metrics_path = os.path.join(output_dir, f"{dataset}_{model_architecture}_metrics{suffix}.npz")
    np.savez(metrics_path, **metrics)
    print(f"\nSaved metrics to: {metrics_path}")
    
    print("\nEvaluation complete!")
    print(f"\nTo load results later:")
    print(f"  data = np.load('{save_path}')")
    print(f"  logits = data['logits']")
    print(f"  features = data['features']")
