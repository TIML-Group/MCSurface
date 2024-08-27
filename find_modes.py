# 90 accuracy  after 40 epoch

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
import matplotlib.pyplot as plt
import os

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

normalize = transforms.Normalize(mean=[0.485, 0.465, 0.406], std=[0.229, 0.224, 0.225])

# Data augmentation and normalization for training
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    normalize
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,
                                          shuffle=True, num_workers=4)

testloader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])), 
    batch_size=128, shuffle=False,
    num_workers=4,pin_memory=True,
)

# Initialize the model with no weights
model = vgg16(weights=None)  # Use weights=None to avoid deprecation warnings
# model = vgg16(weights=VGG16_Weights.DEFAULT)
model.classifier[6] = torch.nn.Linear(4096, 10)  # Adjust for CIFAR-10 classes
model = model.to(device)

# Initialize weights function
def initialize_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

# Apply weight initialization
model.apply(initialize_weights)

# Loss, optimizer, and scheduler
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=5e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

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

epoch_num = list()
loss_num = list()
# Training loop
def train_model():
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
        
        scheduler.step()  # Adjust the learning rate

        # Save checkpoint and log accuracy
        if epoch % 10 == 9:
            accuracy = calculate_accuracy(testloader, model)
            print(f'Epoch {epoch+1} - Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')
            path = './checkpoints3'
            if not os.path.exists(path):
                os.makedirs(path)
            # if epoch > 230 or accuracy > 90:
            if accuracy >= 90:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': average_loss,
                    'accuracy': accuracy
                }, f'{path}/vgg16_cifar10_epoch_{epoch+1}.pth')

train_model()

# paint the loss r.w.t epoch number
plt.figure(figsize=(10, 6))
plt.plot(epoch_num, loss_num, marker='o', linestyle='-', color='b', label='Average Loss')

plt.xlabel('epoch num')
plt.ylabel('avg loss')
plt.grid(True)
plt.legend()

plt.savefig('loss_finding_modes.png', dpi=300, bbox_inches='tight')
plt.show()