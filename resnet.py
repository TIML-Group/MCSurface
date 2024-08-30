# import torch
# import torch.nn as nn
# from torchvision.models import resnet18

# class ResNetNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ResNetNet, self).__init__()
#         # Initialize a standard ResNet-18 model
#         self.resnet = resnet18(pretrained=False)
#         # Replace the fully connected layer to match the number of classes
#         self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

#         # Freeze the parameters in the convolutional layers
#         for param in self.resnet.parameters():
#             param.requires_grad = False

#         # Optionally, you can unfreeze the fully connected layers
#         for param in self.resnet.fc.parameters():
#             param.requires_grad = True

#     def forward(self, x, flat_params=None):
#         # If flat_params is provided, update the parameters of the fc layer
#         if flat_params is not None:
#             self.update_params(flat_params)
#         return self.resnet(x)

#     def update_params(self, flat_params):
#         # Update the weights and biases of the fc layer from flat_params
#         num_features = self.resnet.fc.in_features
#         num_classes = self.resnet.fc.out_features
#         # Calculate sizes of weights and biases
#         weight_size = num_features * num_classes
#         bias_size = num_classes
#         # Update fc layer's weights and biases
#         self.resnet.fc.weight.data.copy_(flat_params[:weight_size].view(num_classes, num_features))
#         self.resnet.fc.bias.data.copy_(flat_params[weight_size:weight_size+bias_size])
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class ResNetNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetNet, self).__init__()
        resnet = resnet18(pretrained=False)
        self.num_classes = num_classes
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

        # Extracting all ResNet layers explicitly
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = resnet.fc

        # Freeze all parameters initially
        for param in self.parameters():
            param.requires_grad = False

    def apply_conv(self, conv, x, flat_params, offset):
        weight_size = conv.weight.numel()
        bias = None
        if conv.bias is not None:
            bias_size = conv.bias.numel()
            bias = flat_params[offset + weight_size:offset + weight_size + bias_size].view_as(conv.bias)
        weight = flat_params[offset:offset + weight_size].view_as(conv.weight)
        offset += weight_size + (bias.numel() if bias is not None else 0)
        return F.conv2d(x, weight, bias, conv.stride, conv.padding, conv.dilation, conv.groups), offset

    def apply_bn(self, bn, x, flat_params, offset):
        num_features = bn.weight.numel()
        weight = flat_params[offset:offset + num_features].view_as(bn.weight)
        bias = flat_params[offset + num_features:offset + 2 * num_features].view_as(bn.bias)
        running_mean = bn.running_mean
        running_var = bn.running_var
        offset += 2 * num_features
        return F.batch_norm(x, running_mean, running_var, weight, bias, bn.training, bn.momentum, bn.eps), offset

    def apply_downsample(self, downsample, x, flat_params, offset):
        for module in downsample.children():
            if isinstance(module, nn.Conv2d):
                x, offset = self.apply_conv(module, x, flat_params, offset)
            elif isinstance(module, nn.BatchNorm2d):
                x, offset = self.apply_bn(module, x, flat_params, offset)
        return x, offset

    def apply_layer(self, layer, x, flat_params, offset):
        for block in layer:
            identity = x
            out, offset = self.apply_conv(block.conv1, x, flat_params, offset)
            out, offset = self.apply_bn(block.bn1, out, flat_params, offset)
            out = F.relu(out)
            out, offset = self.apply_conv(block.conv2, out, flat_params, offset)
            out, offset = self.apply_bn(block.bn2, out, flat_params, offset)

            if block.downsample is not None:
                identity, offset = self.apply_downsample(block.downsample, x, flat_params, offset)

            x = F.relu(out + identity)
        return x, offset

    def forward(self, x, flat_params):
        offset = 0
        x, offset = self.apply_conv(self.conv1, x, flat_params, offset)
        x, offset = self.apply_bn(self.bn1, x, flat_params, offset)
        x = self.relu(x)
        x = self.maxpool(x)

        x, offset = self.apply_layer(self.layer1, x, flat_params, offset)
        x, offset = self.apply_layer(self.layer2, x, flat_params, offset)
        x, offset = self.apply_layer(self.layer3, x, flat_params, offset)
        x, offset = self.apply_layer(self.layer4, x, flat_params, offset)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # Apply fully connected layer
        fc_weight_size = self.fc.weight.numel()
        fc_weight = flat_params[offset:offset + fc_weight_size].view_as(self.fc.weight)
        fc_bias = flat_params[offset + fc_weight_size:offset + fc_weight_size + self.fc.bias.numel()].view_as(self.fc.bias)
        x = F.linear(x, fc_weight, fc_bias)

        return x


# # Example usage
# model = ResNetNet(num_classes=10)
# print(model)

# # Example of loading model
# checkpoint_path = 'checkpoints/Resnet_CIFAR10_run4/model_epoch_350.pth'
# checkpoint = torch.load(checkpoint_path)
# model.load_state_dict(checkpoint['model_state_dict'], strict=True)  # Using strict=False to avoid errors if only partial state should be loaded
# print(model)
