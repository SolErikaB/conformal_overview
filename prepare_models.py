"""
Script which fine-tunes ImageNet pretrained models on CIFAR-10, CIFAR-100, and Fashion-MNIST,
or simply loads the ImageNet model for ImageNet (2012) classification. 
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import os
from tqdm import tqdm
import json
from datetime import datetime
import yaml

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_transforms(dataset_name):
    """Get appropriate transforms for each dataset"""
    
    if dataset_name == 'cifar10':
        # CIFAR-10 specific normalization
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                               std=[0.2470, 0.2435, 0.2616])
        ])
        
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                               std=[0.2470, 0.2435, 0.2616])
        ])
        
    elif dataset_name == 'cifar100':
        # CIFAR-100 specific normalization
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                               std=[0.2675, 0.2565, 0.2761])
        ])
        
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                               std=[0.2675, 0.2565, 0.2761])
        ])
        
    elif dataset_name == 'fashionmnist':
        # Fashion-MNIST: convert to RGB
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                               std=[0.5, 0.5, 0.5])
        ])
        
        transform_val = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                               std=[0.5, 0.5, 0.5])
        ])

    elif dataset_name == 'imagenet':
        # ImageNet normalization - CRITICAL for pretrained models!
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return transform_train, transform_val

def get_datasets(dataset_name, data_root, seed):
    
    from torch.utils.data import ConcatDataset
    
    transform_train, transform_val = get_transforms(dataset_name)
    
    if dataset_name == 'cifar10':
        # CIFAR-10: 32x32 RGB images, 10 classes   
        full_train = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        official_test = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_val)
        
        # Split train into train/val/extra_test (30k/10k/10k)
        train_set, val_set, extra_test_set = random_split(
            full_train, [30000, 10000, 10000], 
            generator=torch.Generator().manual_seed(seed)
        )
        
        # Apply val transforms to val and extra_test sets
        val_full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_val)
        val_set.dataset = val_full
        extra_test_set.dataset = val_full
        
        # Combine the 10k extra_test with 10k official test = 20k test set
        test_set = ConcatDataset([extra_test_set, official_test])
        
        num_classes = 10
        input_size = 32
        
    elif dataset_name == 'cifar100':
        # CIFAR-100: 32x32 RGB images, 100 classes
        full_train = datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
        official_test = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform_val)

        train_set, val_set, extra_test_set = random_split(
            full_train, [30000, 10000, 10000], 
            generator=torch.Generator().manual_seed(seed)
        )
        
        val_full = datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_val)
        val_set.dataset = val_full
        extra_test_set.dataset = val_full
        
        test_set = ConcatDataset([extra_test_set, official_test])
        
        num_classes = 100
        input_size = 32
        
    elif dataset_name == 'fashionmnist':
        # Fashion-MNIST: 28x28 grayscale images, 10 classes
        full_train = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform_train)
        official_test = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform_val)
        
        # For Fashion-MNIST: 60k total, so 30k/20k/10k split
        train_set, val_set, extra_test_set = random_split(
            full_train, [30000, 20000, 10000], 
            generator=torch.Generator().manual_seed(seed)
        )
        
        val_full = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform_val)
        val_set.dataset = val_full
        extra_test_set.dataset = val_full

        # Combine 10k extra_test with 10k official test = 20k test set
        test_set = ConcatDataset([extra_test_set, official_test])

        num_classes = 10
        input_size = 28

    elif dataset_name == 'imagenet':
        # ImageNet: 224x224 RGB images, 1000 classes
        # This requires downloading the ImageNet 2012 dataset separately
        train_set = None
        val_set = datasets.ImageNet(
            root=os.path.join(data_root, 'imagenet'),
            split="val",
            transform=transform_val
        )

        test_set = val_set
        
        num_classes = 1000
        input_size = 224  # Standard for ImageNet
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_set, val_set, test_set, num_classes, input_size

def get_model(arch_name, dataset_name, num_classes, input_size):
    """Load pretrained ImageNet model and adapt for dataset"""
    
    if arch_name == 'resnet18':
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Only replace classifier if NOT ImageNet
        if dataset_name != 'imagenet':
            base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        
    elif arch_name == 'resnet50':
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        if dataset_name != 'imagenet':
            base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        
    elif arch_name == 'efficientnet_b0':
        base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        if dataset_name != 'imagenet':
            in_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features, num_classes)
            )
    
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")
    
    # Wrap with upsample layer if input is small
    if input_size < 224:
        class ModelWithUpsample(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
                self.base_model = base_model
            
            def forward(self, x):
                x = self.upsample(x)
                return self.base_model(x)
        
        model = ModelWithUpsample(base_model)
    else:
        model = base_model
    
    return model

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({'loss': total_loss/total, 'acc': 100.*correct/total})
    
    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, device):
    """Evaluate on validation/test set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

def train_model(arch_name, dataset_name, data_root, model_root, seed, epochs, batch_size, learning_rate, weight_decay, num_workers):
    """Fine-tune a model on a dataset"""
    
    print(f"\n{'='*80}")
    print(f"Training {arch_name} on {dataset_name}")
    print(f"{'='*80}\n")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_set, val_set, test_set, num_classes, input_size = get_datasets(dataset_name, data_root, seed)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers, 
                            pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, 
                        shuffle=False, num_workers=num_workers, 
                        pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers, 
                            pin_memory=True)
    
    # Load model
    model = get_model(arch_name, dataset_name, num_classes, input_size)
    model = model.to(device)
    
    print(f"Model uses pretrained ImageNet weights with upsampling from {input_size}x{input_size} to 224x224")
    
    # Setup training with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Use SGD for better results
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'config': {
            'architecture': arch_name,
            'dataset': dataset_name,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'seed': seed,
            'num_classes': num_classes,
            'input_size': input_size,
            'optimizer': 'SGD',
            'scheduler': 'CosineAnnealingLR',
            'label_smoothing': 0.1,
        }
    }
    
    # Train for fixed number of epochs
    best_val_acc = 0
    best_epoch = 0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Step the scheduler
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            print(f"*** New best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch} ***")
    
    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    print(f"Best Val Acc: {best_val_acc:.2f}% (epoch {best_epoch})")
    
    history['test_loss'] = test_loss
    history['test_acc'] = test_acc
    history['best_val_acc'] = best_val_acc
    history['best_val_epoch'] = best_epoch
    
    # Save model and history
    model_dir = os.path.join(model_root, dataset_name)
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, f"{arch_name}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': history['config'],
        'epoch': epochs,
        'best_val_acc': best_val_acc,
        'best_val_epoch': best_epoch,
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    return history


if __name__ == '__main__':
    
    # Fixed seed for reproducibility
    seed = 123
    set_seed(seed)

    # Load config file with system-specific parameters not important for reproducibility
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    data_directory = config['training']['data_directory']
    model_directory = config['training']['model_directory']

    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(model_directory, exist_ok=True)

    # Training hyperparameters, fine-tuning
    epochs = 50
    batch_size = 128
    learning_rate = 0.01
    weight_decay = 5e-4
    num_workers = config['training']['num_workers']

    # Get dataset and model from config
    dataset = config['training']['dataset']
    model_architecture = config['training']['model_architecture']

    # Training
    print(f"Starting training with seed={seed}")
    print(f"Dataset: {dataset}")
    print(f"Architecture: {model_architecture}")
    print(f"Using pretrained ImageNet weights with upsampling")
    print(f"Optimizer: SGD with momentum=0.9")
    print(f"Scheduler: CosineAnnealingLR")
    print(f"Label smoothing: 0.1")
    
    start_time = datetime.now()
    
    all_results = {}

    try:
        history = train_model(
            arch_name=model_architecture,
            dataset_name=dataset,
            data_root=data_directory,
            model_root=model_directory,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_workers=num_workers
        )
        
        if dataset not in all_results:
            all_results[dataset] = {}
        
        all_results[dataset][model_architecture] = {
            'test_acc': history['test_acc'],
            'best_val_acc': history['best_val_acc'],
            'best_val_epoch': history['best_val_epoch']
        }
    except Exception as e:
        print(f"\nError training {model_architecture} on {dataset}: {e}")
        import traceback
        traceback.print_exc()
        
        if dataset not in all_results:
            all_results[dataset] = {}
        all_results[dataset][model_architecture] = {'error': str(e)}

    # Save summary
    summary_path = os.path.join(model_directory, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'results': all_results,
            'config': {
                'seed': seed,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'optimizer': 'SGD',
                'scheduler': 'CosineAnnealingLR',
                'label_smoothing': 0.1,
                'timestamp': start_time.isoformat(),
            }
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nSummary saved to: {summary_path}")
    print("\nResults:")
    for dataset_name, archs in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        for arch, result in archs.items():
            if 'error' in result:
                print(f"  {arch}: ERROR - {result['error']}")
            else:
                print(f"  {arch}: Test Acc = {result['test_acc']:.2f}%, "
                    f"Best Val Acc = {result['best_val_acc']:.2f}% (epoch {result['best_val_epoch']})")