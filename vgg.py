import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

Debug = False

class VGGNet(nn.Module):
    def __init__(self, num_classes=10):
        super(VGGNet, self).__init__()
        vgg = vgg16(weights=None)
        self.features = vgg.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        for param in self.features.parameters():
            param.requires_grad = False

        # Manually define the classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, num_classes)
        )

        for param in self.classifier.parameters():
            param.requires_grad = False

        self.fc1_in_features = 512 * 7 * 7
        self.fc1_out_features = 4096
        self.fc2_out_features = 4096
        self.fc3_out_features = num_classes

    def forward(self, x, flat_params):
        offset = 0
        if Debug:
            print(f"flat_params total size: {flat_params.numel()}")

        # Manually extracting and using parameters from flat_params for feature layers
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                weight_size = layer.weight.numel()
                bias_size = layer.bias.numel()
                if Debug:
                    print(f"Conv2d layer: weight_size={weight_size}, bias_size={bias_size}")
                weight = flat_params[offset:offset + weight_size].view_as(layer.weight)
                offset += weight_size
                bias = flat_params[offset:offset + bias_size].view_as(layer.bias)
                offset += bias_size
                if Debug:
                    print(f"Updated offset: {offset}")
                x = F.conv2d(x, weight, bias, layer.stride, layer.padding, layer.dilation, layer.groups)
            elif isinstance(layer, nn.ReLU):
                x = F.relu(x, inplace=layer.inplace)
            elif isinstance(layer, nn.MaxPool2d):
                x = F.max_pool2d(x, layer.kernel_size, layer.stride, layer.padding, layer.dilation, layer.ceil_mode)
            elif isinstance(layer, nn.AdaptiveAvgPool2d):
                x = F.adaptive_avg_pool2d(x, layer.output_size)

        # Continue with avgpool
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Classifier layers parameters
        fc1_weight_size = self.fc1_out_features * self.fc1_in_features
        fc1_weight = flat_params[offset:offset + fc1_weight_size].view(self.fc1_out_features, self.fc1_in_features)
        offset += fc1_weight_size
        if Debug:
            print(f"fc1_weight_size: {fc1_weight_size}, offset after fc1_weight: {offset}")

        fc1_bias = flat_params[offset:offset + self.fc1_out_features]
        offset += self.fc1_out_features
        if Debug:
            print(f"fc1_out_features: {self.fc1_out_features}, offset after fc1_bias: {offset}")

        fc2_weight_size = self.fc2_out_features * self.fc1_out_features
        fc2_weight = flat_params[offset:offset + fc2_weight_size].view(self.fc2_out_features, self.fc1_out_features)
        offset += fc2_weight_size
        if Debug:
            print(f"fc2_weight_size: {fc2_weight_size}, offset after fc2_weight: {offset}")

        fc2_bias = flat_params[offset:offset + self.fc2_out_features]
        offset += self.fc2_out_features
        if Debug:
            print(f"fc2_out_features: {self.fc2_out_features}, offset after fc2_bias: {offset}")

        fc3_weight_size = self.fc3_out_features * self.fc2_out_features
        fc3_weight = flat_params[offset:offset + fc3_weight_size].view(self.fc3_out_features, self.fc2_out_features)
        offset += fc3_weight_size
        if Debug:
            print(f"fc3_weight_size: {fc3_weight_size}, offset after fc3_weight: {offset}")

        fc3_bias = flat_params[offset:offset + self.fc3_out_features]
        offset += self.fc3_out_features
        if Debug:
            print(f"fc3_out_features: {self.fc3_out_features}, final offset: {offset}")

        # Forward pass using custom parameters
        x = F.linear(x, fc1_weight, fc1_bias)
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.linear(x, fc2_weight, fc2_bias)
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.linear(x, fc3_weight, fc3_bias)

        return x
