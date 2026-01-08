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
            
            # Features up to layer3 (second-to-last feature layer)
            self.features = nn.Sequential(*children[:-1])
            self.classifier = children[-1]  # fc
            
        elif 'efficientnet' in arch_name:
            # EfficientNet structure: features (Sequential), avgpool, classifier
            children = list(base_model.children())

            # Features up to and including avgpool
            self.features = nn.Sequential(*children[:-1])
            self.classifier = children[-1]  # classifier
            
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


def save_results(results, save_dir, dataset, model_architecture):
    """Save evaluation results to disk"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save main results
    save_path = os.path.join(save_dir, f"{dataset}_{model_architecture}_outputs.npz")
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
    print(f"Top 1 Accuracy: {top_1_accuracy:.4f} ({top_1_accuracy*100:.2f}%)")
    print(f"Top 5 Accuracy: {top_5_accuracy:.4f} ({top_5_accuracy*100:.2f}%)")
    print(f"Features shape: {results['features'].shape}")
    print("="*50)
    
    return {
        'top_1_accuracy': top_1_accuracy,
        'top_5_accuracy': top_5_accuracy,
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
    
    if dataset != 'imagenet':
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