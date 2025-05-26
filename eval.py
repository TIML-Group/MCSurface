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
import yaml

from surfaces import SurfaceNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_surface(u_vals, v_vals, z_vals, title, file_name, value_label):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create a fine grid by interpolating if the data is coarse
    u_vals, v_vals = np.meshgrid(u_vals, v_vals)
    
    # Contour plot with a more sophisticated colormap and more levels
    contour = ax.contourf(u_vals, v_vals, z_vals, levels=100, cmap='viridis', alpha=0.9)
    
    # Add contour lines with labels
    contour_lines = ax.contour(u_vals, v_vals, z_vals, levels=10, colors='black', linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")

    # Highlighting the minimum point on the contour plot
    min_idx = np.unravel_index(np.argmin(z_vals, axis=None), z_vals.shape)
    ax.plot(u_vals[min_idx], v_vals[min_idx], 'ro', markersize=10)
    ax.text(u_vals[min_idx], v_vals[min_idx], 'Min', color='white', fontsize=12, ha='right', va='bottom', weight='bold')

    # Highlighting the maximum point on the contour plot
    max_idx = np.unravel_index(np.argmax(z_vals, axis=None), z_vals.shape)
    ax.plot(u_vals[max_idx], v_vals[max_idx], 'bo', markersize=10)
    ax.text(u_vals[max_idx], v_vals[max_idx], 'Max', color='white', fontsize=12, ha='right', va='top', weight='bold')

    # Add labels and title with enhanced fonts and positioning
    ax.set_xlabel('Parameter u', fontsize=14, labelpad=10)
    ax.set_ylabel('Parameter v', fontsize=14, labelpad=10)
    ax.set_title(title, fontsize=16, pad=20, weight='bold')

    # Add a color bar with more descriptive labeling
    cbar = fig.colorbar(contour)
    cbar.set_label(value_label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Customize the tick parameters for better readability
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Save the figure with a tight layout
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.close()

# def plot_surface(u_vals, v_vals, z_vals, title, file_name):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     u_vals, v_vals = np.meshgrid(u_vals, v_vals)
#     ax.plot_surface(u_vals, v_vals, z_vals, cmap='viridis')
#     ax.set_xlabel('u')
#     ax.set_ylabel('v')
#     ax.set_zlabel('Value')
#     ax.set_title(title)
#     plt.savefig(file_name)
#     plt.close()

def inspect_curves(surface_net, test_loader):
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
    parser = argparse.ArgumentParser(description="Evaluate Bezier SurfaceNet from a saved model and plot surfaces.")
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file.')

    # Arguments that can be in YAML or overridden by CLI
    parser.add_argument('--model_type', choices=['Vgg', 'Resnet', 'Vit', 'MobileNet', 'SimpleCNN'], help='Base model architecture used for SurfaceNet')
    parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100', 'TinyImageNet'], help='Dataset used for SurfaceNet')
    parser.add_argument('--num_bends', type=int, help='Number of bends for the Bezier surface')
    parser.add_argument('--checkpoint_paths', nargs='+', help='Paths to the four base model checkpoint files (needed for SurfaceNet instantiation if not all params saved)')
    parser.add_argument('--surface_model_path', type=str, help='Path to the saved SurfaceNet .pth model file')
    
    # Eval specific arguments (less likely to be in training config, but can be)
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for evaluation DataLoader')


    # Temporary parse to find the config file path
    temp_args, _ = parser.parse_known_args()
    config_params = {}
    if temp_args.config:
        with open(temp_args.config, 'r') as f:
            config_params = yaml.safe_load(f)
            if config_params is None: # Handle empty YAML file
                config_params = {}
        # print(f"Loaded config for eval from {temp_args.config}: {config_params}") # For debugging

    # Set defaults from YAML config before final parsing
    # We need to ensure that only args expected by SurfaceNet or eval.py are passed from training config
    # For example, learning_rate, init_epochs etc. from training config are not needed here.
    # We can selectively pull them or let argparse handle it if names match.
    # For clarity, explicitly set defaults for arguments `eval.py` defines.

    if 'model_type' in config_params:
        parser.set_defaults(model_type=config_params['model_type'])
    if 'dataset' in config_params:
        parser.set_defaults(dataset=config_params['dataset'])
    if 'num_bends' in config_params:
        parser.set_defaults(num_bends=config_params['num_bends'])
    if 'checkpoint_paths' in config_params: # these are still needed to initialize SurfaceNet correctly
        parser.set_defaults(checkpoint_paths=config_params['checkpoint_paths'])
    
    # If output_surface_model_path_template is in config, try to construct the surface_model_path
    if 'output_surface_model_path_template' in config_params and 'surface_model_path' not in config_params:
        try:
            # Ensure all necessary keys for formatting are present in config_params
            model_type_cfg = config_params.get('model_type')
            dataset_cfg = config_params.get('dataset')
            num_bends_cfg = config_params.get('num_bends')
            if model_type_cfg and dataset_cfg and num_bends_cfg is not None:
                 constructed_path = config_params['output_surface_model_path_template'].format(
                    model_type=model_type_cfg,
                    dataset=dataset_cfg,
                    num_bends=num_bends_cfg
                )
                 parser.set_defaults(surface_model_path=constructed_path)
                 print(f"Constructed surface_model_path from template: {constructed_path}") # Debug
            else:
                print("Warning: Could not construct surface_model_path from template due to missing parameters in config.")
        except KeyError as e:
            print(f"Warning: Key {e} not found in config when trying to format output_surface_model_path_template.")

    if 'surface_model_path' in config_params: # Direct path from config takes precedence
         parser.set_defaults(surface_model_path=config_params['surface_model_path'])


    args = parser.parse_args()

    # Check if essential arguments are provided either via CLI or config
    if not args.model_type:
        parser.error("the following arguments are required: --model_type (or provide in config)")
    if not args.dataset:
        parser.error("the following arguments are required: --dataset (or provide in config)")
    if not args.num_bends:
        parser.error("the following arguments are required: --num_bends (or provide in config)")
    if not args.surface_model_path:
        parser.error("the following arguments are required: --surface_model_path (or provide in config or parsable from output_surface_model_path_template in config)")
    if not args.checkpoint_paths: # Still needed for SurfaceNet init
        parser.error("the following arguments are required: --checkpoint_paths (or provide in config)")


    # Default values if not in args or config (though most are required now)
    # learning_rate, num_samples, init_epochs, total_epochs are not used in eval.py directly for SurfaceNet
    # weight_decay is needed for SurfaceNet constructor, provide a default or make it configurable.
    # Using a placeholder, as it's part of SurfaceNet's signature.
    # It does not affect evaluation if we are just loading state_dict and calling eval methods.
    placeholder_lr = 0.001 
    placeholder_wd = 0.0
    placeholder_num_samples = 1
    placeholder_init_epochs = 1
    placeholder_total_epochs = 1


    num_classes_map = {
        'CIFAR10': 10,
        'CIFAR100': 100,
        'TinyImageNet': 200
    }

    # Transformations - should match training.
    # These are common defaults; if training used different ones, this section might need to be configurable too.
    if args.dataset == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        image_size = 32
    elif args.dataset == 'CIFAR100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        image_size = 32
    elif args.dataset == 'TinyImageNet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_size = 64
    else: # Should not happen due to choices in argparse
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # ViT specific transforms
    if args.model_type == 'Vit':
         transform_test = transforms.Compose([
            transforms.Resize(224), 
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform_test = transforms.Compose([
            transforms.Resize(image_size), # Ensure resize for non-ViT models if image_size is not native
            transforms.ToTensor(),
            normalize
        ])


    # Load dataset
    if args.dataset == 'CIFAR10':
        # train_dataset needed for SurfaceNet constructor, but its content not critical for eval if loading state
        train_dataset_placeholder = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test) 
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif args.dataset == 'CIFAR100':
        train_dataset_placeholder = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_test)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    elif args.dataset == 'TinyImageNet':
        # Assuming TinyImageNet is stored in './data/tiny-imagenet-200'
        train_dataset_placeholder = datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform_test)
        test_dataset = datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=transform_test) # Ensure val set for testing


    # For demo purposes, allow subsetting, though not typically from training config
    # test_dataset = Subset(test_dataset, range(1024)) # Example
    # train_dataset_placeholder = Subset(train_dataset_placeholder, range(10)) # Minimal placeholder

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    surface_net = SurfaceNet(
        num_classes=num_classes_map[args.dataset],
        model_type=args.model_type,
        num_bends=args.num_bends,
        learning_rate=placeholder_lr, # Placeholder
        weight_decay=placeholder_wd,  # Placeholder
        num_samples=placeholder_num_samples, # Placeholder
        dataset=train_dataset_placeholder, # Placeholder dataset instance
        batch_size=args.batch_size, # Can use eval batch_size
        init_epochs=placeholder_init_epochs, # Placeholder
        total_epochs=placeholder_total_epochs, # Placeholder
        checkpoint_paths=args.checkpoint_paths # Critical for initializing theta structure
        # parameter_dropout_rate is part of SurfaceNet, if it was added, set to 0 for eval
        # parameter_dropout_rate=0.0 
    ).to(device=device)


    print(f"Loading SurfaceNet model from: {args.surface_model_path}")
    surface_net.load_state_dict(torch.load(args.surface_model_path, map_location=device), strict=True)
    surface_net.eval()

    # If inspect_curves is to be used:
    # inspect_curves(surface_net, test_loader) 

    u_vals, v_vals, loss_surface, accuracy_surface = surface_net.evaluate_on_grid(test_loader)

    # Find the minimum loss and maximum accuracy
    # Ensure surfaces are not all NaNs (e.g., if test_loader was empty or evaluation failed)
    if not np.all(np.isnan(loss_surface)):
        min_loss = np.nanmin(loss_surface)
        min_loss_idx = np.unravel_index(np.nanargmin(loss_surface), loss_surface.shape)
        min_loss_u = u_vals[min_loss_idx[0]]
        min_loss_v = v_vals[min_loss_idx[1]]
        print(f"Lowest loss value: {min_loss:.4f} at u={min_loss_u:.2f}, v={min_loss_v:.2f}")
    else:
        print("Loss surface contains all NaN values.")
        min_loss = float('nan')

    if not np.all(np.isnan(accuracy_surface)):
        max_accuracy = np.nanmax(accuracy_surface)
        max_accuracy_idx = np.unravel_index(np.nanargmax(accuracy_surface), accuracy_surface.shape)
        max_accuracy_u = u_vals[max_accuracy_idx[0]]
        max_accuracy_v = v_vals[max_accuracy_idx[1]]
        print(f"Highest accuracy: {max_accuracy:.2f}% at u={max_accuracy_u:.2f}, v={max_accuracy_v:.2f}")

    else:
        print("Accuracy surface contains all NaN values.")
        max_accuracy = float('nan')


    # Ensure plot directory exists
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    loss_plot_filename = os.path.join(plot_dir, f"loss_surface_{args.model_type}_{args.dataset}_b{args.num_bends}.png")
    accuracy_plot_filename = os.path.join(plot_dir, f"accuracy_surface_{args.model_type}_{args.dataset}_b{args.num_bends}.png")

    plot_surface(u_vals, v_vals, loss_surface, f"Loss Surface ({args.model_type} on {args.dataset})", loss_plot_filename, "Loss")
    plot_surface(u_vals, v_vals, accuracy_surface, f"Accuracy Surface ({args.model_type} on {args.dataset})", accuracy_plot_filename, "Accuracy (%)")

    print(f"Plots saved to {plot_dir}/")
