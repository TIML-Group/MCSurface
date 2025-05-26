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
        self.num_classes = num_classes

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

    def _apply_conv(self, conv, x, flat_params, offset):
        weight_size = conv.weight.numel()
        bias_size = conv.bias.numel()
        weight = flat_params[offset:offset + weight_size].view_as(conv.weight)
        offset += weight_size
        bias = flat_params[offset:offset + bias_size].view_as(conv.bias)
        offset += bias_size
        return F.conv2d(x, weight, bias, conv.stride, conv.padding, conv.dilation, conv.groups), offset

    def _apply_bn(self, bn, x, flat_params, offset):
        weight_size = bn.weight.numel()
        bias_size = bn.bias.numel()
        mean_var_size = bn.weight.numel()
        weight = flat_params[offset:offset + weight_size].view_as(bn.weight)
        offset += weight_size
        bias = flat_params[offset:offset + bias_size].view_as(bn.bias)
        offset += bias_size
        running_mean_flat = flat_params[offset:offset + mean_var_size].view_as(bn.running_mean).detach()
        offset += mean_var_size
        running_var_flat = flat_params[offset:offset + mean_var_size].view_as(bn.running_var).detach()
        offset += mean_var_size
        return F.batch_norm(x, running_mean_flat, running_var_flat, weight, bias, bn.training, bn.momentum, bn.eps), offset

    def _apply_linear(self, linear, x, flat_params, offset):
        weight_size = linear.weight.numel()
        bias_size = linear.bias.numel()
        weight = flat_params[offset:offset + weight_size].view_as(linear.weight)
        offset += weight_size
        bias = flat_params[offset:offset + bias_size].view_as(linear.bias)
        offset += bias_size
        return F.linear(x, weight, bias), offset

    def forward(self, x, flat_params):
        offset = 0
        if Debug:
            print(f"flat_params total size: {flat_params.numel()}")

        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                x, offset = self._apply_conv(layer, x, flat_params, offset)
            elif isinstance(layer, nn.BatchNorm2d):
                x, offset = self._apply_bn(layer, x, flat_params, offset)
            elif isinstance(layer, nn.ReLU):
                x = F.relu(x, inplace=layer.inplace)
            elif isinstance(layer, nn.MaxPool2d):
                x = F.max_pool2d(x, layer.kernel_size, layer.stride, layer.padding, layer.dilation, layer.ceil_mode)
            elif isinstance(layer, nn.AdaptiveAvgPool2d):
                x = F.adaptive_avg_pool2d(x, layer.output_size)

        # Continue with avgpool
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Classifier layers
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                x, offset = self._apply_linear(layer, x, flat_params, offset)
            elif isinstance(layer, nn.ReLU):
                x = F.relu(x, inplace=layer.inplace)
            elif isinstance(layer, nn.Dropout):
                x = F.dropout(x, p=layer.p, training=self.training)

        return x
