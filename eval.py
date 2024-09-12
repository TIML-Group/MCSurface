import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader, Subset
from torchvision.models import vgg16
from scipy.special import binom
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

from surfaces import SurfaceNet
import vgg


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_surface(u_vals, v_vals, z_vals, title, file_name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    u_vals, v_vals = np.meshgrid(u_vals, v_vals)
    ax.plot_surface(u_vals, v_vals, z_vals, cmap='viridis')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_zlabel('Value')
    ax.set_title(title)
    plt.savefig(file_name)
    plt.close()

def inspect_curves():
    with torch.no_grad():
        surface_net.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        v = torch.rand(1).to(device)
        # for _ in range(self.num_samples):
        u = torch.rand(1).to(device)
        points = surface_net.compute_training_points(u, v)
        point = points[1]

        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            # for point in points:
            output = surface_net.net(x, point)  # Pass the point as flat_params to self.net
            loss = surface_net.compute_loss(output, y) / len(points)

            total_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            correct_predictions += (preds == y).sum().item()
            total_samples += y.size(0)

        average_loss = total_loss / total_samples
        accuracy = (correct_predictions / total_samples) * 100
        print(accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['Vgg', 'Resnet', 'Vit'], default='Vgg', help='Choose the model to use')
    parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100', 'TinyImageNet'], default='CIFAR10', help='Choose the dataset')
    args = parser.parse_args()

    num_bends = 2
    learning_rate = 0.0002
    num_samples = 5
    batch_size = 512
    init_epochs = 10
    total_epochs = 20


    # specify classes number to help with building up model configuration
    num_classes = {
        'CIFAR10': 10,
        'CIFAR100': 100,
        'TinyImageNet': 200
    }

    # Checkpoint paths
    # checkpoint_paths = [
    #     './checkpoints1/vgg16_cifar10_epoch_280.pth',
    #     './checkpoints2/vgg16_cifar10_epoch_280.pth',
    #     './checkpoints3/vgg16_cifar10_epoch_340.pth',
    #     './checkpoints4/vgg16_cifar10_epoch_280.pth'
    # ]
    # checkpoint_paths = [
    #     './checkpoints/vgg16_cifar10_epoch_70.pth',
    #     './checkpoints/vgg16_cifar10_epoch_80.pth',
    #     './checkpoints/vgg16_cifar10_epoch_90.pth',
    #     './checkpoints/vgg16_cifar10_epoch_100.pth'
    # ]

    checkpoint_paths = [
        'checkpoints/Resnet_CIFAR100_run1/model_epoch_360.pth',
        'checkpoints/Resnet_CIFAR100_run2/model_epoch_360.pth',
        'checkpoints/Resnet_CIFAR100_run3/model_epoch_360.pth',
        'checkpoints/Resnet_CIFAR100_run4/model_epoch_360.pth'
    ]

    # checkpoint_paths = [
    #     'checkpoints/Vgg_CIFAR10_run1/model_epoch_350.pth',
    #     'checkpoints/Vgg_CIFAR10_run2/model_epoch_350.pth',
    #     'checkpoints/Vgg_CIFAR10_run3/model_epoch_350.pth',
    #     'checkpoints/Vgg_CIFAR10_run4/model_epoch_350.pth'
    # ]

    # Transformations for CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.465, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

    # for demo purpose only take first 5000
    # train_dataset = Subset(train_dataset, range(1024))

    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    surface_net = SurfaceNet(num_classes[args.dataset], args.model, num_bends, learning_rate, num_samples, train_dataset, batch_size, init_epochs, total_epochs, checkpoint_paths)
    # surface_net.load_state_dict(torch.load('SurfaceNet.pth'), strict=True)
    # surface_net.load_state_dict(torch.load('Surface_cifar100.pth'), strict=True)

    u_vals, v_vals, loss_surface, accuracy_surface = surface_net.evaluate_on_grid(test_loader)
    plot_surface(u_vals, v_vals, loss_surface, "Loss Surface", "loss_surface.png")
    plot_surface(u_vals, v_vals, accuracy_surface, "Accuracy Surface", "accuracy_surface.png")
