import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class ResNetNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetNet, self).__init__()
        # Use weights=None for modern torchvision, or pretrained=False for older versions
        # For consistency with find_modes.py, let's assume weights=None is appropriate.
        resnet = resnet18(weights=None) 
        self.num_classes = num_classes
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

        # Store layers from the loaded resnet model
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = nn.ReLU(inplace=True) # Use nn.ReLU instance for self.relu(x)
        self.maxpool = resnet.maxpool     # Use the maxpool layer from resnet directly
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool   # Use the avgpool layer from resnet directly
        self.fc = resnet.fc

        # Removing the loop that sets requires_grad = False for all parameters.
        # This allows normal training and weight loading if the model is used standalone.
        # set_model_parameters in surfaces.py will set them as needed.

    # --- Helper methods for forward pass with flat_params --- 
    def _apply_conv(self, conv, x, flat_params, offset):
        weight_size = conv.weight.numel()
        current_bias = None
        bias_size = 0
        if conv.bias is not None:
            bias_size = conv.bias.numel()
            current_bias = flat_params[offset + weight_size : offset + weight_size + bias_size].view_as(conv.bias)
        weight = flat_params[offset : offset + weight_size].view_as(conv.weight)
        offset += weight_size + bias_size
        return F.conv2d(x, weight, current_bias, conv.stride, conv.padding, conv.dilation, conv.groups), offset

    def _apply_bn(self, bn, x, flat_params, offset):
        weight_size = bn.weight.numel()
        bias_size = bn.bias.numel()
        # running_mean and running_var have the same numel as weight/bias for BN
        mean_var_size = bn.weight.numel() # Each is num_features

        weight = flat_params[offset : offset + weight_size].view_as(bn.weight)
        offset += weight_size
        bias = flat_params[offset : offset + bias_size].view_as(bn.bias)
        offset += bias_size
        
        # Consume running_mean from flat_params - DETACH this for F.batch_norm
        running_mean_flat = flat_params[offset : offset + mean_var_size].view_as(bn.running_mean).detach()
        offset += mean_var_size
        
        # Consume running_var from flat_params - DETACH this for F.batch_norm
        running_var_flat = flat_params[offset : offset + mean_var_size].view_as(bn.running_var).detach()
        offset += mean_var_size
        
        # weight and bias RETAIN their original requires_grad status from flat_params
        return F.batch_norm(x, 
                            running_mean_flat,  # Detached
                            running_var_flat,   # Detached
                            weight,             # Original (potentially requires_grad)
                            bias,               # Original (potentially requires_grad)
                            bn.training, bn.momentum, bn.eps), offset

    def _apply_downsample(self, downsample, x, flat_params, offset):
        # Assuming downsample is a Sequential module containing Conv2d and BatchNorm2d
        for module_name, module in downsample.named_children():
            if isinstance(module, nn.Conv2d):
                x, offset = self._apply_conv(module, x, flat_params, offset)
            elif isinstance(module, nn.BatchNorm2d):
                x, offset = self._apply_bn(module, x, flat_params, offset)
            # Add other module types if present in downsample
        return x, offset

    def _apply_layer(self, layer, x, flat_params, offset):
        # layer is a Sequential of BasicBlock
        for block_name, block in layer.named_children(): # Iterate through blocks in the layer
            identity = x
            out, offset = self._apply_conv(block.conv1, x, flat_params, offset)
            out, offset = self._apply_bn(block.bn1, out, flat_params, offset)
            out = self.relu(out) # Use self.relu as it's an nn.Module
            out, offset = self._apply_conv(block.conv2, out, flat_params, offset)
            out, offset = self._apply_bn(block.bn2, out, flat_params, offset)

            if block.downsample is not None:
                identity, offset = self._apply_downsample(block.downsample, identity, flat_params, offset) # Pass identity to downsample

            out += identity
            x = self.relu(out) # Apply relu after adding identity
        return x, offset

    def forward(self, x, flat_params):
        offset = 0
        x, offset = self._apply_conv(self.conv1, x, flat_params, offset)
        x, offset = self._apply_bn(self.bn1, x, flat_params, offset)
        x = self.relu(x)
        x = self.maxpool(x)

        x, offset = self._apply_layer(self.layer1, x, flat_params, offset)
        x, offset = self._apply_layer(self.layer2, x, flat_params, offset)
        x, offset = self._apply_layer(self.layer3, x, flat_params, offset)
        x, offset = self._apply_layer(self.layer4, x, flat_params, offset)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        fc_weight_size = self.fc.weight.numel()
        fc_bias_size = 0
        if self.fc.bias is not None:
            fc_bias_size = self.fc.bias.numel()
        
        fc_weight = flat_params[offset : offset + fc_weight_size].view_as(self.fc.weight)
        offset += fc_weight_size
        
        current_fc_bias = None
        if self.fc.bias is not None:
            current_fc_bias = flat_params[offset : offset + fc_bias_size].view_as(self.fc.bias)
        
        x = F.linear(x, fc_weight, current_fc_bias)
        return x

# # Example usage
# model = ResNetNet(num_classes=10)
# print(model)

# # Example of loading model
# checkpoint_path = 'checkpoints/Resnet_CIFAR10_run4/model_epoch_350.pth'
# checkpoint = torch.load(checkpoint_path)
# model.load_state_dict(checkpoint['model_state_dict'], strict=True)  # Using strict=False to avoid errors if only partial state should be loaded
# print(model)
 # This should be False if model is in eval mode
                           