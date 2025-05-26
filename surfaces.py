import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader, Subset
from torchvision.models import vgg16, resnet18
from scipy.special import binom
import yaml # Added YAML import
import os # Ensure os is imported for path operations

# Import VGGNet from vgg.py
from vgg import VGGNet
from resnet import ResNetNet
from vit import VitNet
from mobilenet import MobileNetNet
from simplecnn import SimpleCNNNet

Debug = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_parameters(model):
    params_list = []
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            params_list.append(module.weight.flatten())
            if module.bias is not None:
                params_list.append(module.bias.flatten())
        elif isinstance(module, torch.nn.Linear):
            params_list.append(module.weight.flatten())
            if module.bias is not None:
                params_list.append(module.bias.flatten())
        elif isinstance(module, torch.nn.BatchNorm2d):
            params_list.append(module.weight.flatten())
            params_list.append(module.bias.flatten())
            params_list.append(module.running_mean.flatten())  # Buffers, but that's OK
            params_list.append(module.running_var.flatten())   # Buffers, but that's OK
    if not params_list:
        print("Warning: params_list is empty in get_model_parameters. Model might be empty or structured unexpectedly.")
        return torch.tensor([]).to(device)
    flat_params_with_bn_buffers = torch.cat(params_list)
    if Debug:
        print(f"Total elements in flat_params (including BN buffers) from get_model_parameters: {flat_params_with_bn_buffers.numel()}")
    return flat_params_with_bn_buffers

# def get_model_parameters(model):
#     params_list = []
#     # Ensure model is on the correct device before extracting parameters/buffers
#     # model.to(device) # Or assume it's already on the device

#     # Iterate through modules to maintain order and identify BN layers
#     # The order of parameters/buffers added here MUST match the consumption
#     # order in ResNetNet's _apply_conv, _apply_bn, etc.
#     for module_name, module in model.named_modules():
#         if isinstance(module, torch.nn.Conv2d):
#             params_list.append(module.weight.data.flatten()) # Use .data to get tensor without grad_fn
#             if module.bias is not None:
#                 params_list.append(module.bias.data.flatten())
#         elif isinstance(module, torch.nn.Linear):
#             params_list.append(module.weight.data.flatten())
#             if module.bias is not None:
#                 params_list.append(module.bias.data.flatten())
#         elif isinstance(module, torch.nn.BatchNorm2d):
#             # Order for ResNetNet._apply_bn: weight, bias, running_mean, running_var
#             params_list.append(module.weight.data.flatten())
#             params_list.append(module.bias.data.flatten())
#             params_list.append(module.running_mean.data.flatten()) # it's a buffer, .data is fine
#             params_list.append(module.running_var.data.flatten())  # it's a buffer, .data is fine
#             # num_batches_tracked is also a buffer, but not used by F.batch_norm's core calculation
#             # and not typically included in such flattened parameter vectors unless specifically needed.

#     if not params_list:
#         # This might happen if the model has no Conv, Linear, or BN layers,
#         # or if named_modules() iteration doesn't pick them up as expected.
#         # Fallback or error for empty model? For ResNet, this shouldn't be empty.
#         print("Warning: params_list is empty in get_model_parameters. Model might be empty or structured unexpectedly.")
#         return torch.tensor([]).to(device)

#     # Concatenate all collected tensors
#     # Ensure all tensors are on the same device before concatenation if not already handled
#     # Assuming all parts of the model were moved to `device` before calling this function.
#     flat_params_with_bn_buffers = torch.cat(params_list)
    
#     if Debug:
#         print(f"Total elements in flat_params (including BN buffers) from get_model_parameters: {flat_params_with_bn_buffers.numel()}")
#     return flat_params_with_bn_buffers

def print_model_parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    if Debug:
        print(f"Expected total number of parameters: {total_params}")

class BezierSurface(Module):
    def __init__(self, n, m):
        super().__init__()
        self.register_buffer('n', torch.tensor(n, dtype=torch.float32))
        self.register_buffer('m', torch.tensor(m, dtype=torch.float32))
        self.register_buffer('binom_n', torch.Tensor(binom(n, np.arange(n + 1), dtype=np.float32)))
        self.register_buffer('binom_m', torch.Tensor(binom(m, np.arange(m + 1), dtype=np.float32)))
        self.register_buffer('range_n', torch.arange(0, float(n + 1)))
        self.register_buffer('rev_range_n', torch.arange(float(n), -1, -1))
        self.register_buffer('range_m', torch.arange(0, float(m + 1)))
        self.register_buffer('rev_range_m', torch.arange(float(m), -1, -1))

    def forward(self, u, v):
        B_u = self.binom_n * torch.pow(u, self.range_n) * torch.pow((1.0 - u), self.rev_range_n)
        B_v = self.binom_m * torch.pow(v, self.range_m) * torch.pow((1.0 - v), self.rev_range_m)
        return B_u, B_v

class SurfaceNet(Module):
    def __init__(self, num_classes, model_type, num_bends, learning_rate, weight_decay, num_samples, dataset, batch_size, init_epochs, total_epochs, checkpoint_paths):
        super(SurfaceNet, self).__init__()
        self.num_classes = num_classes
        self.model_type = model_type
        self.num_bends = num_bends
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay  # Added weight_decay
        self.num_samples = num_samples
        self.dataset = dataset
        self.batch_size = batch_size
        self.init_epochs = init_epochs
        self.total_epochs = total_epochs

        self.bezier_surface = BezierSurface(num_bends, num_bends).to(device)
        self.theta = self.initialize_control_points(checkpoint_paths)
        self.net = self.create_net_model()  # Initialize the custom model

        print_model_parameter_count(self.net)  # Add this line to debug

    def initialize_control_points(self, checkpoint_paths):
        theta = torch.nn.ParameterDict()

        # Load models from checkpoints
        models = [self.load_model_from_checkpoint(path) for path in checkpoint_paths]

        # Set fixed endpoints
        theta['0,0'] = Parameter(get_model_parameters(models[0]), requires_grad=False)
        theta[f'0,{self.num_bends}'] = Parameter(get_model_parameters(models[1]), requires_grad=False)
        theta[f'{self.num_bends},0'] = Parameter(get_model_parameters(models[2]), requires_grad=False)
        theta[f'{self.num_bends},{self.num_bends}'] = Parameter(get_model_parameters(models[3]), requires_grad=False)

        # Initialize other control points using linear interpolation
        for i in range(self.num_bends + 1):
            for j in range(self.num_bends + 1):
                if (i, j) not in [(0, 0), (0, self.num_bends), (self.num_bends, 0), (self.num_bends, self.num_bends)]:
                    t = i / self.num_bends
                    s = j / self.num_bends
                    theta[f'{i},{j}'] = Parameter(
                        (1 - t) * (1 - s) * theta['0,0'].data +
                        t * (1 - s) * theta[f'{self.num_bends},0'].data +
                        (1 - t) * s * theta[f'0,{self.num_bends}'].data +
                        t * s * theta[f'{self.num_bends},{self.num_bends}'].data,
                        requires_grad=True
                    )
        return theta

    def load_model_from_checkpoint(self, checkpoint_path):
        model = self.create_net_model()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        # model.load_state_dict(checkpoint, strict=True)
        model.to(device)
        return model

    def get_model_parameters(self, model):
        # Flatten and concatenate all parameters of the model into a single tensor
        return torch.cat([p.flatten().to(device) for p in model.parameters()])

    def set_model_parameters(self, model, parameters):
        device = next(model.parameters()).device
        # Reshape and set the parameters for the model
        offset = 0
        for param in model.parameters():
            param_size = param.numel()
            param.data.copy_(parameters[offset:offset + param_size].reshape(param.size()).to(device))
            offset += param_size

    def create_net_model(self):
        if self.model_type == 'Vgg':
            return VGGNet(num_classes=self.num_classes).to(device)
        elif self.model_type == 'Resnet':
            return ResNetNet(num_classes=self.num_classes).to(device)
        elif self.model_type == 'Vit':
            return VitNet(num_classes=self.num_classes).to(device)
        elif self.model_type == 'MobileNet':
            return MobileNetNet(num_classes=self.num_classes).to(device)
        elif self.model_type == 'SimpleCNN':
            return SimpleCNNNet(num_classes=self.num_classes).to(device)

    def compute_initialization_points(self, u, v):
        B_u, B_v = self.bezier_surface(u, v)
        B1 = sum(self.theta[f'0,{j}'] * B_v[j] for j in range(self.num_bends + 1))
        B2 = sum(self.theta[f'{self.num_bends},{j}'] * B_v[j] for j in range(self.num_bends + 1))
        return B1, B2

    def compute_training_points(self, u, v):
        B_u, B_v = self.bezier_surface(u, v)
        points = []
        for j in range(self.num_bends + 1):
            B_j = sum(self.theta[f'{i},{j}'] * B_v[i] for i in range(self.num_bends + 1))
            points.append(B_j)
        return points

    def forward(self, u, v):
        B_u, B_v = self.bezier_surface(u, v)
        B_surface = {}
        for i in range(self.num_bends + 1):
            for j in range(self.num_bends + 1):
                B_surface[f'{i},{j}'] = self.theta[f'{i},{j}'] * B_u[i] * B_v[j]
        return B_surface

    def create_model(self):
        # Create a model with the same architecture as the one from the checkpoints
        if self.model_type == 'Vgg':
            model = vgg16(weights=None)
            model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, self.num_classes)
            return model
        else:
            model = resnet18(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, self.num_classes)
            return model

    def train_model(self):
        self.to(device)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        
        # Stage 1: Initialization phase
        for epoch in range(self.init_epochs):
            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            for x, y in DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=2):
                x, y = x.to(device), y.to(device)
                
                for _ in range(self.num_samples):
                    u = torch.rand(1).to(device)
                    v = torch.rand(1).to(device)
                    B1, B2 = self.compute_initialization_points(u, v)
                    
                    output1 = self.net(x, B1)  # Pass B1 as flat_params to self.net
                    output2 = self.net(x, B2)  # Pass B2 as flat_params to self.net
                    
                    loss = (self.compute_loss(output1, y) + self.compute_loss(output2, y)) / 2
                    optimizer.zero_grad()
                    loss.backward()

                    if Debug:
                        for name, param in self.named_parameters():
                            if param.requires_grad:
                                if param.grad is None:
                                    print(f"{name} gradient is None")
                                else:
                                    print(f"{name}: {param.grad.norm().item()}")
                    
                    optimizer.step()

                    total_loss += loss.item()
                    preds1 = torch.argmax(output1, dim=1)
                    preds2 = torch.argmax(output2, dim=1)
                    correct_predictions += ((preds1 == y).sum().item() + (preds2 == y).sum().item()) / 2
                    total_samples += y.size(0)

                    del B1, B2, output1, output2, loss
                    torch.cuda.empty_cache()

            average_loss = total_loss / total_samples
            accuracy = (correct_predictions / total_samples) * 100
            print(f"Initialization Epoch {epoch + 1}/{self.init_epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")
            torch.cuda.empty_cache()

        # Stage 3: Generalization Phase

        for epoch in range(self.init_epochs, self.total_epochs):
            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            for x, y in DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4):
                x, y = x.to(device), y.to(device)
                
                for _ in range(self.num_samples):
                    u = torch.rand(1).to(device)
                    v = torch.rand(1).to(device)

                    B_surface = self.forward(u, v)
                    params = sum(B_surface[f'{m},{n}'] for m in range(self.num_bends + 1) for n in range(self.num_bends + 1))
                    
                    output = self.net(x, params)  # Pass params as flat_params to self.net
                    
                    loss = self.compute_loss(output, y)
                    optimizer.zero_grad()
                    loss.backward()

                    if Debug:
                        for name, param in self.named_parameters():
                            if param.requires_grad:
                                if param.grad is None:
                                    print(f"{name} gradient is None")
                                else:
                                    print(f"{name}: {param.grad.norm().item()}")
                    
                    optimizer.step()

                    total_loss += loss.item()
                    preds = torch.argmax(output, dim=1)
                    correct_predictions += (preds == y).sum().item()
                    total_samples += y.size(0)

                    del B_surface, output, loss
                    torch.cuda.empty_cache()

            average_loss = total_loss / total_samples
            accuracy = (correct_predictions / total_samples) * 100
            print(f"Generalization Epoch {epoch + 1}/{self.total_epochs - self.init_epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

    def compute_loss(self, output, target):
        return F.cross_entropy(output, target)

    def ensemble_evaluation(surface_net, test_loader, interval=0.1):
        surface_net.eval()  # Set the model to evaluation mode
        surface_net.to(device)
        u_vals = np.arange(0, 1.0 + interval, interval)
        v_vals = np.arange(0, 1.0 + interval, interval)
        num_samples = len(u_vals) * len(v_vals)
        criterion = torch.nn.CrossEntropyLoss()
        model = surface_net.create_model().to(device)
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                batch_size = x.size(0)
                ensemble_outputs = torch.zeros(batch_size, surface_net.num_classes).to(device)
                
                for u in u_vals:
                    for v in v_vals:
                        u_tensor = torch.tensor([u], dtype=torch.float32).to(device)
                        v_tensor = torch.tensor([v], dtype=torch.float32).to(device)
                        
                        # Compute Bezier surface parameters
                        B_surface = surface_net.forward(u_tensor, v_tensor)
                        params = sum(B_surface[f'{m},{n}'] for m in range(surface_net.num_bends + 1) for n in range(surface_net.num_bends + 1))
                        
                        # Set model parameters
                        surface_net.set_model_parameters(model, params)
                        
                        # Get model output
                        output = model(x)
                        probs = torch.softmax(output, dim=1)
                        
                        # Aggregate outputs
                        ensemble_outputs += probs
                    
                # Average the outputs
                ensemble_outputs /= num_samples
                
                # Compute loss
                loss = criterion(ensemble_outputs.log(), y)
                total_loss += loss.item() * batch_size
                
                # Compute predictions
                preds = ensemble_outputs.argmax(dim=1)
                correct_predictions += (preds == y).sum().item()
                total_samples += batch_size
        
        average_loss = total_loss / total_samples
        accuracy = (correct_predictions / total_samples) * 100
        print(f"Ensemble Test - Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")

    def evaluate_on_grid(self, test_loader, interval=0.1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()  # Set the model to evaluation mode
        u_vals = np.arange(0, 1 + interval, interval)
        v_vals = np.arange(0, 1 + interval, interval)
        loss_surface = np.zeros((len(u_vals), len(v_vals)))
        accuracy_surface = np.zeros((len(u_vals), len(v_vals)))
        model = self.create_net_model().to(device)
        with torch.no_grad():
            for i, u in enumerate(u_vals):
                for j, v in enumerate(v_vals):
                    u_tensor = torch.tensor([u], dtype=torch.float32).to(device)
                    v_tensor = torch.tensor([v], dtype=torch.float32).to(device)
                    B_u, B_v = self.bezier_surface(u_tensor, v_tensor)
                    B_surface = self.forward(u_tensor, v_tensor)
                    
                    total_loss = 0
                    correct_predictions = 0
                    total_samples = 0

                    # params = sum(B_surface[f'{m},{n}'] for m in range(self.num_bends + 1) for n in range(self.num_bends + 1))
                    # params = torch.zeros_like(B_surface['0,0'].data, device=device)
                    # for m in range(self.num_bends + 1):
                    #     for n in range(self.num_bends + 1):
                    #         params += B_surface[f'{m},{n}']

                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)

                        params = torch.zeros_like(B_surface['0,0'].data, device=device)
                        for m in range(self.num_bends + 1):
                            for n in range(self.num_bends + 1):
                                params += B_surface[f'{m},{n}']

                        output = model(x, params)
                        loss = self.compute_loss(output, y)
                        total_loss += loss.item()
                        preds = torch.argmax(output, dim=1)
                        correct_predictions += (preds == y).sum().item()
                        total_samples += y.size(0)

                    if total_samples > 0:
                        loss_surface[i, j] = total_loss / total_samples
                        current_accuracy = (correct_predictions / total_samples) * 100
                        accuracy_surface[i, j] = current_accuracy

                        total_loss_pt = 0 
                    else:
                        loss_surface[i, j] = float('nan') # Handle case with no samples

        return u_vals, v_vals, loss_surface, accuracy_surface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bezier SurfaceNet using settings from a YAML config file or command-line arguments.")
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file for SurfaceNet.')

    # --- Arguments that can be in YAML or overridden by CLI ---
    parser.add_argument('--model_type', choices=['Vgg', 'Resnet', 'Vit', 'MobileNet', 'SimpleCNN'], help='Choose the base model architecture')
    parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100', 'TinyImageNet'], help='Choose the dataset')
    parser.add_argument('--checkpoint_paths', nargs='+', help='Paths to the four base model checkpoint files')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the SurfaceNet optimizer')
    parser.add_argument('--weight_decay', type=float, help='Weight decay for the SurfaceNet optimizer')
    parser.add_argument('--num_bends', type=int, help='Number of bends for the Bezier surface')
    parser.add_argument('--num_samples', type=int, help='Number of samples for SurfaceNet training iterations')
    parser.add_argument('--batch_size', type=int, help='Batch size for SurfaceNet training DataLoader')
    parser.add_argument('--init_epochs', type=int, help='Number of epochs for initialization phase')
    parser.add_argument('--total_epochs', type=int, help='Number of epochs for main training phase (before generalization)')
    parser.add_argument('--generalization_additional_epochs', type=int, help='Additional epochs for generalization phase')
    parser.add_argument('--output_surface_model_path_template', type=str, help='Template for the output surface model path')

    # Temporary parse to find the config file path
    temp_args, _ = parser.parse_known_args()
    config_params = {}
    if temp_args.config:
        with open(temp_args.config, 'r') as f:
            config_params = yaml.safe_load(f)
            # print(f"Loaded config from {temp_args.config}: {config_params}") # For debugging

    # Set defaults from YAML config before final parsing
    parser.set_defaults(**config_params)
    args = parser.parse_args()

    # num_bends = 3
    # learning_rate = args.lr
    # weight_decay = args.weight_decay
    # num_samples = 15
    # batch_size = 256

    # Specify number of classes for each dataset
    num_classes_map = {
        'CIFAR10': 10,
        'CIFAR100': 100,
        'TinyImageNet': 200
    }
    # model_type = args.model
    # init_epochs = 6
    # total_epochs = 15

    # checkpoint_paths = args.checkpoints

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
    if args.model_type == 'Vit':
        # ViT requires 224x224 images
        transform_train = transforms.Compose([
            transforms.Resize(224),  # Resize images to 224x224 for ViT
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomGrayscale(p=0.1),
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
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            normalize
        ])

        transform_test = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize
        ])

    # Load dataset
    if args.dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    elif args.dataset == 'CIFAR100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    elif args.dataset == 'TinyImageNet':
        # Assuming TinyImageNet is stored in './data/tiny-imagenet-200'
        train_dataset = datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform_train)

    # Checkpoint paths
    # args.checkpoint_paths = [
    # checkpoint_paths = [
    #     './checkpoints1/vgg16_cifar10_epoch_280.pth',
    #     './checkpoints2/vgg16_cifar10_epoch_280.pth',
    #     './checkpoints3/vgg16_cifar10_epoch_340.pth',
    #     './checkpoints4/vgg16_cifar10_epoch_280.pth'
    # ]


    checkpoint_paths = args.checkpoint_paths  # Use checkpoint paths from arguments

    # checkpoint_paths = [
    #     './new_checkpoints/Resnet_CIFAR10_run1/model_epoch_220.pth',
    #     './new_checkpoints/Resnet_CIFAR10_run2/model_epoch_220.pth',
    #     './new_checkpoints/Resnet_CIFAR10_run3/model_epoch_220.pth',
    #     './new_checkpoints/Resnet_CIFAR10_run4/model_epoch_220.pth'
    # ]

    # checkpoint_paths = [
    #     'checkpoints/Resnet_CIFAR100_run1/model_epoch_360.pth',
    #     'checkpoints/Resnet_CIFAR100_run2/model_epoch_360.pth',
    #     'checkpoints/Resnet_CIFAR100_run3/model_epoch_360.pth',
    #     'checkpoints/Resnet_CIFAR100_run4/model_epoch_360.pth'
    # ]

    # args.checkpoints = [
    #     'checkpoints/Resnet_CIFAR100_run1/model_epoch_360.pth',
    #     'checkpoints/Resnet_CIFAR100_run2/model_epoch_360.pth',
    #     'checkpoints/Resnet_CIFAR100_run3/model_epoch_360.pth',
    #     'checkpoints/Resnet_CIFAR100_run4/model_epoch_360.pth'
    # ]

    # checkpoint_paths = [
    #     'checkpoints/Resnet_CIFAR10_run1/model_epoch_350.pth',
    #     'checkpoints/Resnet_CIFAR10_run2/model_epoch_350.pth',
    #     'checkpoints/Resnet_CIFAR10_run3/model_epoch_350.pth',
    #     'checkpoints/Resnet_CIFAR10_run4/model_epoch_350.pth'
    # ]

    # checkpoint_paths = [
    #     'checkpoints/SimpleCNN_CIFAR10_run1/model_epoch_20.pth',
    #     'checkpoints/SimpleCNN_CIFAR10_run2/model_epoch_20.pth',
    #     'checkpoints/SimpleCNN_CIFAR10_run3/model_epoch_20.pth',
    #     'checkpoints/SimpleCNN_CIFAR10_run4/model_epoch_20.pth'
    # ]

    # checkpoint_paths = [
    #     'checkpoints/MobileNet_CIFAR10_run1/model_epoch_20.pth',
    #     'checkpoints/MobileNet_CIFAR10_run2/model_epoch_20.pth',
    #     'checkpoints/MobileNet_CIFAR10_run3/model_epoch_20.pth',
    #     'checkpoints/MobileNet_CIFAR10_run4/model_epoch_20.pth'
    # ]

    # checkpoint_paths = [
    #     'checkpoints/Vit_CIFAR10_run1/model_epoch_350.pth',
    #     'checkpoints/Vit_CIFAR10_run1/model_epoch_360.pth',
    #     'checkpoints/Vit_CIFAR10_run1/model_epoch_350.pth',
    #     'checkpoints/Vit_CIFAR10_run1/model_epoch_360.pth',
    # ]

    # For demo purposes only, take a subset of the data
    # train_dataset = Subset(train_dataset, range(128))
    # train_dataset = Subset(train_dataset, range(512))
    # train_dataset = Subset(train_dataset, range(8192))

    surface_net = SurfaceNet(
        num_classes=num_classes_map[args.dataset],
        model_type=args.model_type,
        num_bends=args.num_bends,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_samples=args.num_samples,
        dataset=train_dataset,
        batch_size=args.batch_size,
        init_epochs=args.init_epochs,
        total_epochs=args.total_epochs, # This is the main training phase duration
        checkpoint_paths=checkpoint_paths
    )
    # Note: The generalization phase in SurfaceNet.train_model adds more epochs.
    # The `generalization_additional_epochs` from config could be passed to `train_model` if needed
    # or used to adjust the loop in `train_model` directly.
    # For now, the structure of train_model regarding generalization epochs is kept as is.

    # compile to speed up
    # surface_net = torch.compile(surface_net)

    surface_net.train_model() # The generalization phase inside adds args.generalization_additional_epochs
    
    # Construct the output path from the template
    output_filename = args.output_surface_model_path_template.format(
        model_type=args.model_type,
        dataset=args.dataset,
        num_bends=args.num_bends
    )
    # Ensure the directory exists if the template includes a path
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.save(surface_net.state_dict(), output_filename)
    print(f"Saved SurfaceNet model to {output_filename}")

    # Example of loading (ensure path matches what you saved)
    # surface_net.load_state_dict(torch.load(output_filename), strict=False)
