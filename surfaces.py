import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader, Subset
from torchvision.models import vgg16
from scipy.special import binom

# Import VGGNet from vgg.py
from vgg import VGGNet

Debug = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_parameters(model):
    flat_params = torch.cat([p.flatten().to(device) for p in model.parameters()])
    if Debug:
        print(f"Total parameters: {flat_params.numel()}")  # Add this line to debug
    return flat_params

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
    def __init__(self, num_classes, num_bends, learning_rate, num_samples, dataset, batch_size, init_epochs, total_epochs, checkpoint_paths):
        super(SurfaceNet, self).__init__()
        self.num_classes = num_classes
        self.num_bends = num_bends
        self.learning_rate = learning_rate
        self.num_samples = num_samples
        self.dataset = dataset
        self.batch_size = batch_size
        self.init_epochs = init_epochs
        self.total_epochs = total_epochs

        self.bezier_surface = BezierSurface(num_bends, num_bends).to(device)
        self.theta = self.initialize_control_points(checkpoint_paths)
        self.net = self.create_vggnet_model()  # Initialize the custom VGGNet model

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
                        t * s * theta[f'{self.num_bends},{self.num_bends}'].data
                    )
        return theta

    def load_model_from_checkpoint(self, checkpoint_path):
        model = self.create_vggnet_model()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
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

    def create_vggnet_model(self):
        return VGGNet(num_classes=self.num_classes).to(device)

    def compute_initialization_points(self, u, v):
        B_u, B_v = self.bezier_surface(u, v)
        B1 = sum(self.theta[f'0,{j}'] * B_v[j] for j in range(self.num_bends + 1))
        B2 = sum(self.theta[f'{self.num_bends},{j}'] * B_v[j] for j in range(self.num_bends + 1))
        return B1, B2

    def compute_training_points(self, u, v):
        # B_u, B_v = self.bezier_surface(u, v)
        # points = []
        # for i in range(1, self.num_bends):
        #     B_i = sum(self.theta[f'{i},{j}'] * B_u[j] for j in range(self.num_bends + 1))
        #     points.append(B_i)
        # return points
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
        # model = self.architecture(weights=None)
        model = vgg16(weights=None)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, self.num_classes)
        return model
        # model = self.create_vggnet_model().to(device)
        # return model

    def train_model(self):
        self.to(device)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=2e-4)
        
        # Stage 1: Initialization phase
        for epoch in range(self.init_epochs):
            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            for x, y in DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=2):
                x, y = x.to(device), y.to(device)
                
                for _ in range(self.num_samples):
                    u = torch.rand(1).to(device)
                    # for _ in range(self.num_samples):
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
            # print(f"Initialization Epoch {epoch + 1}/{self.init_epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")
            print(f"Initialization Epoch {epoch + 1}/{self.init_epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Stage 2: Training phase
        for epoch in range(self.init_epochs, self.total_epochs):
            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            for x, y in DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True):
                x, y = x.to(device), y.to(device)
                
                for _ in range(self.num_samples):
                    v = torch.rand(1).to(device)
                    # for _ in range(self.num_samples):
                    u = torch.rand(1).to(device)
                    points = self.compute_training_points(u, v)
                    
                    for point in points:
                        output = self.net(x, point)  # Pass the point as flat_params to self.net
                        loss = self.compute_loss(output, y) / len(points)

                        optimizer.zero_grad()
                        loss.backward()
                        
                        # Ensure fixed endpoints are not updated
                        for i in [0, self.num_bends]:
                            for j in range(self.num_bends + 1):
                                self.theta[f'{i},{j}'].grad = None

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

                        del point, output, loss  # Free up memory
                        torch.cuda.empty_cache()

            average_loss = total_loss / total_samples
            accuracy = (correct_predictions / total_samples) * 100
            # print(f"Training Epoch {epoch + 1 - self.init_epochs}/{self.total_epochs - self.init_epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")
            print(f"Training Epoch {epoch + 1 - self.init_epochs}/{self.total_epochs - self.init_epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Stage 3: Generalization Phase
        for epoch in range(self.total_epochs, self.total_epochs + 10):
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
            # print(f"Initialization Epoch {epoch + 1}/{self.init_epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")
            print(f"Generalization Epoch {epoch + 1}/{self.init_epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

    def compute_loss(self, output, target):
        return F.cross_entropy(output, target)

    def evaluate_on_grid(self, test_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()  # Set the model to evaluation mode
        self.to(device)
        u_vals = np.arange(0, 1.1, 0.2)
        v_vals = np.arange(0, 1.1, 0.2)
        loss_surface = np.zeros((len(u_vals), len(v_vals)))
        accuracy_surface = np.zeros((len(u_vals), len(v_vals)))

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

                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)
                        model = self.create_model().to(device)
                        params = sum(B_surface[f'{m},{n}'] for m in range(self.num_bends + 1) for n in range(self.num_bends + 1))
                        self.set_model_parameters(model, params)
                        output = model(x)
                        loss = self.compute_loss(output, y)
                        total_loss += loss.item()
                        preds = torch.argmax(output, dim=1)
                        correct_predictions += (preds == y).sum().item()
                        total_samples += y.size(0)

                    loss_surface[i, j] = total_loss / total_samples
                    accuracy_surface[i, j] = (correct_predictions / total_samples) * 100

        return u_vals, v_vals, loss_surface, accuracy_surface

# Example usage
if __name__ == "__main__":
    num_bends = 2
    learning_rate = 0.0001
    num_samples = 15
    # num_samples = 1
    batch_size = 512
    num_classes = 10
    # init_epochs = 10
    # total_epochs = 20
    init_epochs = 6
    total_epochs = 15

    # Checkpoint paths
    checkpoint_paths = [
        './checkpoints1/vgg16_cifar10_epoch_280.pth',
        './checkpoints2/vgg16_cifar10_epoch_280.pth',
        './checkpoints3/vgg16_cifar10_epoch_340.pth',
        './checkpoints4/vgg16_cifar10_epoch_280.pth'
    ]
    # checkpoint_paths = [
    #     './checkpoints4/vgg16_cifar10_epoch_210.pth',
    #     './checkpoints4/vgg16_cifar10_epoch_230.pth',
    #     './checkpoints4/vgg16_cifar10_epoch_250.pth',
    #     './checkpoints4/vgg16_cifar10_epoch_280.pth'
    # ]

    # Transformations for CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.465, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # for demo purpose only take first 5000
    # train_dataset = Subset(train_dataset, range(512))

    surface_net = SurfaceNet(num_classes, num_bends, learning_rate, num_samples, train_dataset, batch_size, init_epochs, total_epochs, checkpoint_paths)
    surface_net.train_model()
    torch.save(surface_net.state_dict(), f'SurfaceNet.pth')
