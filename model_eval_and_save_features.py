import prepare_models
import yaml
import os
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
            
            # Features up to layer4 (second-to-last feature layer)
            self.features_early = nn.Sequential(*children[:-2])  # Up to layer4
            self.features_late = children[-2]  # avgpool
            self.classifier = children[-1]  # fc
            
        elif 'efficientnet' in arch_name:
            # EfficientNet structure: features (Sequential), avgpool, classifier
            children = list(base_model.children())
            
            # The features module contains all conv blocks
            feature_blocks = list(children[0].children())
            
            # Split: all blocks except last, then last block
            self.features_early = nn.Sequential(*feature_blocks[:-1])
            self.features_last_block = feature_blocks[-1]
            self.features_late = children[1]  # avgpool
            self.classifier = children[2]  # classifier
            
        else:
            raise ValueError(f"Unsupported architecture: {arch_name}")
    
    def forward(self, x, return_features=False):
        # Apply upsample if needed
        if self.has_upsample:
            x = self.upsample(x)
        
        # Extract features from second-to-last layer
        if 'resnet' in self.arch_name:
            feat_early = self.features_early(x)  # After layer4
            feat_late = self.features_late(feat_early)  # After avgpool
            feat_late = feat_late.flatten(1)
            
            # Also flatten the early features for consistency
            feat_early = torch.nn.functional.adaptive_avg_pool2d(feat_early, (1, 1))
            feat_early = feat_early.flatten(1)
            
        elif 'efficientnet' in self.arch_name:
            feat_early = self.features_early(x)
            feat_early = self.features_last_block(feat_early)  # After last conv block
            feat_late = self.features_late(feat_early)  # After avgpool
            feat_late = feat_late.flatten(1)
            
            # Flatten early features
            feat_early = torch.nn.functional.adaptive_avg_pool2d(feat_early, (1, 1))
            feat_early = feat_early.flatten(1)
        
        # Get logits
        logits = self.classifier(feat_late)
        
        if return_features:
            return logits, feat_early, feat_late
        return logits


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set and extract all outputs.
    
    Returns:
        dict with keys: 'logits', 'probabilities', 'features_penultimate', 
                       'features_pre_penultimate', 'labels', 'predictions'
    """
    model.eval()
    
    all_logits = []
    all_probs = []
    all_features_early = []
    all_features_late = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            
            # Get logits and features from both layers
            logits, features_early, features_late = model(images, return_features=True)
            
            # Compute probabilities (softmax of logits)
            probs = torch.softmax(logits, dim=1)
            
            # Store results
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_features_early.append(features_early.cpu().numpy())
            all_features_late.append(features_late.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Concatenate all batches
    results = {
        'logits': np.concatenate(all_logits, axis=0),
        'probabilities': np.concatenate(all_probs, axis=0),
        'features_pre_penultimate': np.concatenate(all_features_early, axis=0),
        'features_penultimate': np.concatenate(all_features_late, axis=0),
        'labels': np.concatenate(all_labels, axis=0),
    }
    
    # Add predictions
    results['predictions'] = np.argmax(results['logits'], axis=1)
    
    return results


def save_results(results, save_dir, dataset, model_architecture):
    """Save evaluation results to disk"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save main results
    save_path = os.path.join(save_dir, f"{dataset}_{model_architecture}_outputs.npz")
    np.savez_compressed(
        save_path,
        logits=results['logits'],
        probabilities=results['probabilities'],
        features_pre_penultimate=results['features_pre_penultimate'],
        features_penultimate=results['features_penultimate'],
        labels=results['labels'],
        predictions=results['predictions']
    )
    
    print(f"\nSaved outputs to: {save_path}")
    print(f"  - Logits shape: {results['logits'].shape}")
    print(f"  - Probabilities shape: {results['probabilities'].shape}")
    print(f"  - Pre-penultimate features shape: {results['features_pre_penultimate'].shape}")
    print(f"  - Penultimate features shape: {results['features_penultimate'].shape}")
    print(f"  - Labels shape: {results['labels'].shape}")
    
    return save_path


def compute_metrics(results):
    """Compute and display evaluation metrics"""
    predictions = results['predictions']
    labels = results['labels']
    
    # Overall accuracy
    accuracy = (predictions == labels).mean()
    
    # Per-class accuracy
    num_classes = results['logits'].shape[1]
    per_class_acc = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            class_acc = (predictions[mask] == c).mean()
            per_class_acc.append(class_acc)
    
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Number of samples: {len(labels)}")
    print(f"Number of classes: {num_classes}")
    print(f"Mean per-class accuracy: {np.mean(per_class_acc):.4f}")
    print(f"Pre-penultimate feature dimension: {results['features_pre_penultimate'].shape[1]}")
    print(f"Penultimate feature dimension: {results['features_penultimate'].shape[1]}")
    print("="*50)
    
    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'mean_per_class_accuracy': np.mean(per_class_acc)
    }


if __name__ == "__main__":
    
    seed = 123
    prepare_models.set_seed(seed)
    
    # Load config file with system-specific parameters
    with open("config.yaml", 'r') as f:
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
    _, _, test_set, num_classes, input_size = prepare_models.get_datasets(dataset, data_dir, seed)
    print(f"Test set size: {len(test_set)}")
    print(f"Number of classes: {num_classes}")
    print(f"Input size: {input_size}")
    
    # Create test dataloader
    test_loader = DataLoader(
        test_set,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load model
    print(f"\nLoading model: {model_architecture}")
    model = prepare_models.get_model(model_architecture, dataset, num_classes, input_size)
    
    model_path = os.path.join(model_dir, dataset, f"{model_architecture}.pth")
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
    save_path = save_results(results, output_dir, dataset, model_architecture)
    
    # Optional: Save metrics separately
    metrics_path = os.path.join(output_dir, f"{dataset}_{model_architecture}_metrics.npz")
    np.savez(metrics_path, **metrics)
    print(f"\nSaved metrics to: {metrics_path}")
    
    print("\nEvaluation complete!")
    print(f"\nTo load results later:")
    print(f"  data = np.load('{save_path}')")
    print(f"  logits = data['logits']")
    print(f"  features = data['features']")