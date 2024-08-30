import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16, resnet18
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from timm.models import vit_small_patch16_224
import os

# set random seed for reproduction
torch.manual_seed(3407)  # from arXiv all you need is seed(3407)
np.random.seed(0)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse argument
parser = argparse.ArgumentParser(description="Train the modes and store in the specified checkpoint path")
parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'ImageNet'], help='Dataset type')
parser.add_argument('--model', type=str, default='Vgg', choices=['Vgg', 'Resnet', 'Vit'], help='Model type')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--checkpoint_folder', type=str, default='./checkpoints', help='Folder to save checkpoints')
args = parser.parse_args()

normalize = transforms.Normalize(mean=[0.485, 0.465, 0.406], std=[0.229, 0.224, 0.225])

# Data augmentation and normalization for training
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    normalize
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

# specify classes number to help with building up model configuration
num_classes = {
    'CIFAR10': 10,
    'CIFAR100': 100,
    'ImageNet': 1000
}

# Load dataset
if args.data == 'CIFAR10':
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
elif args.data == 'CIFAR100':
    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
elif args.data == 'ImageNet':
    # Placeholder for ImageNet dataset loading, adjust transforms as needed
    pass

# create dataloader
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

# Initialize the model based on type and adjust classifier
if args.model == 'Vgg':
    model = vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, num_classes[args.data])
elif args.model == 'Resnet':
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes[args.data])
elif args.model == 'Vit':
    model = vit_small_patch16_224(pretrained=False, num_classes=num_classes[args.data])

model.to(device)

# Initialize weights function
def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

# Apply weight initialization
model.apply(initialize_weights)

# Loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=360, pct_start=0.3, anneal_strategy='linear')
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)

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
    for epoch in range(360):  # Reduced number of epochs for initial testing
        model.train()  # Set model to training mode
        total_loss = 0.0
        total_batches = 0
        
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
        
        # scheduler.step()  # Adjust the learning rate

        # Validate after each epoch
        val_loss = validate(model, test_loader)
        val_loss_list.append(val_loss)
        scheduler.step(val_loss + 3 * average_loss)  # Adjust learning rate based on validation loss

        # Save checkpoint and log accuracy
        if epoch % 10 == 9:
            model.eval()
            with torch.no_grad():
                accuracy = calculate_accuracy(test_loader, model)
                print(f'Epoch {epoch+1} - Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')
                if not os.path.exists(args.checkpoint_folder):
                    os.makedirs(args.checkpoint_folder)
                if accuracy >= 90 or epoch > 340:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': average_loss,
                        'accuracy': accuracy
                    }, f'{args.checkpoint_folder}/model_epoch_{epoch+1}.pth')

    # paint the loss r.w.t epoch number
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_num, loss_num, marker='o', linestyle='-', color='b', label='Average Loss')
    plt.plot(epoch_num, val_loss_list, marker='x', linestyle='--', color='r', label='Validation Loss')
    plt.xlabel('epoch num')
    plt.ylabel('training loss and val loss')
    plt.grid(True)
    plt.legend()

    plt.savefig('loss_finding_modes.png', dpi=300, bbox_inches='tight')
    # optional show
    # plt.show()

if __name__ == '__main__':
    train_model()
