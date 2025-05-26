import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16, resnet18, MobileNet_V2_Weights
from torch import nn
from torchvision.models import mobilenet_v2
import matplotlib.pyplot as plt
import numpy as np
from timm.models import vit_small_patch16_224
import os
import wandb
import yaml
import requests
import zipfile

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Parse arguments
parser = argparse.ArgumentParser(description="Train models using settings from a YAML config file or command-line arguments.")
parser.add_argument('--config', type=str, help='Path to the YAML configuration file.')

# --- Arguments that can be in YAML or overridden by CLI ---
# These will have their defaults set by YAML if a config file is provided
parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100', 'TinyImageNet'], help='Dataset type')
parser.add_argument('--model', type=str, choices=['Vgg', 'Resnet', 'Vit', 'MobileNet'], help='Model type')
parser.add_argument('--lr', type=float, help='Learning rate')
parser.add_argument('--weight_decay', type=float, help='Weight decay')
parser.add_argument('--num_epochs', type=int, help='Total number of training epochs')
parser.add_argument('--batch_size', type=int, help='Batch size for training and testing')
parser.add_argument('--checkpoint_folder', type=str, help='Folder to save checkpoints')

# Scheduler specific args (example for MultiStepLR)
parser.add_argument('--scheduler_type', type=str, help='Type of LR scheduler (e.g., MultiStepLR)')
parser.add_argument('--scheduler_milestones', type=int, nargs='+', help='Milestones for MultiStepLR scheduler')
parser.add_argument('--scheduler_gamma', type=float, help='Gamma value for MultiStepLR scheduler')

# Wandb specific args
parser.add_argument('--wandb_project', type=str, help='WandB project name')
parser.add_argument('--wandb_name_prefix', type=str, default='run', help='Prefix for WandB run name')


# Initial parsing to get config file path if provided
# We need to parse known args here to avoid errors if other args are present that are not yet defined
# based on the config. This is a bit tricky. A simpler way:
# 1. Define all possible args in parser.
# 2. Load config.
# 3. Set defaults from config.
# 4. Parse all args.

# Temporary parse to find the config file path
temp_args, _ = parser.parse_known_args()
config_params = {}
if temp_args.config:
    with open(temp_args.config, 'r') as f:
        config_params = yaml.safe_load(f)

# Set defaults from YAML config before final parsing
# This allows CLI arguments to override YAML settings.
parser.set_defaults(**config_params)

args = parser.parse_args()

def download_and_extract_tiny_imagenet(data_dir='./data'):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    local_zip_file = os.path.join(data_dir, "tiny-imagenet-200.zip")
    extract_dir = os.path.join(data_dir, "tiny-imagenet-200")

    # Check if the dataset is already downloaded and extracted
    if not os.path.exists(extract_dir):
        print("Tiny ImageNet dataset not found. Downloading...")
        # Download the dataset
        response = requests.get(url, stream=True)
        with open(local_zip_file, 'wb') as f:
            f.write(response.content)
        print("Download complete.")

        # Extract the dataset
        print("Extracting Tiny ImageNet dataset...")
        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete.")
    else:
        print("Tiny ImageNet dataset already exists. Skipping download and extraction.")

# Dataset-specific normalization and image size
if args.dataset == 'CIFAR10':
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    image_size = 32
elif args.dataset == 'CIFAR100':
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    image_size = 32
elif args.dataset == 'TinyImageNet':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_size = 64

# Data augmentation and normalization for training
if args.model == 'Vit':
    # ViT requires 224x224 images
    transform_train = transforms.Compose([
        transforms.Resize(224),  # Resize images to 224x224 for ViT
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),  # Resize images to 224x224 for ViT
        transforms.ToTensor(),
        normalize

    ])
else:
    # Use original image sizes for CIFAR10, CIFAR100, and TinyImageNet for other models
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(image_size, padding=4),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=config_params.get('random_erasing', {}).get('p', 0.5),
            scale=tuple(config_params.get('random_erasing', {}).get('scale', [0.02, 0.25])),
            ratio=tuple(config_params.get('random_erasing', {}).get('ratio', [0.3, 3.3])),
            value=0, inplace=False
        ),
        normalize
    ])

    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize
    ])


# specify number of classes for each dataset
num_classes = {
    'CIFAR10': 10,
    'CIFAR100': 100,
    'TinyImageNet': 200  # Tiny ImageNet has 200 classes
}

# Load dataset
if args.dataset == 'CIFAR10':
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
elif args.dataset == 'CIFAR100':
    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
elif args.dataset == 'TinyImageNet':
    download_and_extract_tiny_imagenet(data_dir='./data')
    train_dir = './data/tiny-imagenet-200/train'
    val_dir = './data/tiny-imagenet-200/val'
    train_set = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
    test_set = torchvision.datasets.ImageFolder(root=val_dir, transform=transform_test)

# overfit
# train_set = torch.utils.data.Subset(train_set, range(512))

# Create dataloaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Initialize the model based on type and adjust classifier
if args.model == 'Vgg':
    model = vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, num_classes[args.dataset])
elif args.model == 'Resnet':
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes[args.dataset])
elif args.model == 'Vit':
    model = vit_small_patch16_224(weights=None, num_classes=num_classes[args.dataset])
elif args.model == 'MobileNet':
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes[args.dataset])

model.to(device)

# Initialize weights function
def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

# Apply weight initialization
model.apply(initialize_weights)

# num_epochs = 230
# num_epochs = args.num_epochs # Use num_epochs from args (config or CLI)
# num_epochs = 21

# Loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss() # Removed label_smoothing
# optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

# Uncomment to use different schedulers
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=6)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
# steplr 40, 0.5 work for vgg/cifar10
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.3)
# steplr 30, 0.5 works for resnet18 on cifar10
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs) # Changed scheduler
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160], gamma=0.2) # Restored milestones for 200 epochs
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70, 120, 160, 180], gamma=0.2) # resenet 18 89 acc
if args.scheduler_type == 'MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler_milestones, gamma=args.scheduler_gamma)
elif args.scheduler_type == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
# Add other schedulers as needed
else: # Default or if not specified
    print(f"Warning: Scheduler type '{args.scheduler_type}' not recognized or not specified. Using default MultiStepLR.")
    # Provide some fallback default if nothing is specified via CLI or YAML for scheduler
    default_milestones = [int(args.num_epochs * 0.5), int(args.num_epochs * 0.75)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.scheduler_milestones if args.scheduler_milestones else default_milestones,
        gamma=args.scheduler_gamma if args.scheduler_gamma else 0.1
    )

# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=len(train_loader) * num_epochs, pct_start=0.3, anneal_strategy='cos')
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=50)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=360)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5)

# Function to calculate accuracy
def calculate_accuracy(loader, model):
    correct = 0
    total = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Function to perform validation
def validate(model, loader):
    model.eval()
    total_val_loss = 0.0
    total_val_batches = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item()
            total_val_batches += 1
    return total_val_loss / total_val_batches

# Training loop
def train_model():
    epoch_num = list()
    loss_num = list()
    val_loss_list = list()
    for epoch in range(args.num_epochs): # Use args.num_epochs
        model.train()  # Set model to training mode
        total_loss = 0.0
        total_batches = 0
        
        current_lr = optimizer.param_groups[0]['lr'] # Get current learning rate

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        average_loss = total_loss / total_batches
        epoch_num.append(epoch + 1)
        loss_num.append(average_loss)
        print(f'Epoch {epoch + 1} - Average Loss: {average_loss:.4f}')
        
        # Validate after each epoch
        val_loss = validate(model, test_loader)
        val_loss_list.append(val_loss)
        print(f'Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}')

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "average_training_loss": average_loss,
            "validation_loss": val_loss,
            "learning_rate": current_lr
        })

        # ReduceLROnPlateau scheduler
        # scheduler.step(val_loss)  # Adjust the learning rate
        # normal scheduler
        scheduler.step()

        # Save checkpoint and log accuracy
        if epoch % 10 == 9:
            model.eval()
            with torch.no_grad():
                accuracy = calculate_accuracy(test_loader, model)
                print(f'Epoch {epoch+1} - Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')
                if not os.path.exists(args.checkpoint_folder):
                    os.makedirs(args.checkpoint_folder)
                
                # Log accuracy to wandb
                wandb.log({"accuracy": accuracy, "epoch": epoch + 1})

                if accuracy >= 90 or epoch > args.num_epochs - 20: # Use args.num_epochs
                    checkpoint_path = f'{args.checkpoint_folder}/model_epoch_{epoch+1}.pth'
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': average_loss,
                        'accuracy': accuracy
                    }, checkpoint_path)
                    # wandb.save(checkpoint_path) # Optional: Save model to wandb

    # Plot the training and validation loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_num, loss_num, marker='o', linestyle='-', color='b', label='Average Loss')
    plt.plot(epoch_num, val_loss_list, marker='x', linestyle='--', color='r', label='Validation Loss')
    plt.xlabel('epoch num')
    plt.ylabel('training loss and val loss')
    plt.grid(True)
    plt.legend()

    plt.savefig('loss_finding_modes.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    # Initialize wandb
    wandb_run_name = f"{args.wandb_name_prefix}_{args.model}_{args.dataset}_lr{args.lr}_wd{args.weight_decay}_ep{args.num_epochs}"
    # Ensure all args used in wandb_run_name have valid values (e.g., from YAML or CLI)
    
    wandb_project_name = args.wandb_project if args.wandb_project else "find_modes_default_project"


    wandb.init(project=wandb_project_name, name=wandb_run_name, config=vars(args))

    train_model()
    wandb.finish() # Finish the wandb run
