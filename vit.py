import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import vit_small_patch16_224
import math

class VitNet(nn.Module):
    def __init__(self, num_classes=10):
        super(VitNet, self).__init__()
        # Instantiate the VisionTransformer directly from the timm library
        vit = vit_small_patch16_224(pretrained=False, num_classes=num_classes)

        # Extract and name each component exactly as in the pretrained model for direct access
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.patch_embed = vit.patch_embed
        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.pre_logits = torch.nn.Identity()
        self.head = vit.head

        # Freeze all parameters initially
        for param in self.parameters():
            param.requires_grad = False

    def apply_layernorm(self, layernorm, x, flat_params, offset):
        weight_size = layernorm.weight.numel()
        bias_size = layernorm.bias.numel()
        weight = flat_params[offset:offset + weight_size].view_as(layernorm.weight)
        offset += weight_size
        bias = flat_params[offset:offset + bias_size].view_as(layernorm.bias)
        offset += bias_size
        return F.layer_norm(x, normalized_shape=layernorm.normalized_shape, weight=weight, bias=bias, eps=layernorm.eps), offset

    def apply_linear(self, x, weight_shape, flat_params, offset):
        weight_size = torch.prod(torch.tensor(weight_shape)).item()
        bias_size = weight_shape[0]
        weight = flat_params[offset:offset + weight_size].view(weight_shape)
        offset += weight_size
        bias = flat_params[offset:offset + bias_size]
        offset += bias_size
        return F.linear(x, weight, bias), offset

    def apply_mlp(self, mlp, x, flat_params, offset):
        # Apply first linear layer
        weight_shape1 = mlp.fc1.weight.shape
        x, offset = self.apply_linear(x, weight_shape1, flat_params, offset)
        x = F.gelu(x)
        x = F.dropout(x, p=mlp.drop1.p, training=self.training)

        # Apply second linear layer
        weight_shape2 = mlp.fc2.weight.shape
        x, offset = self.apply_linear(x, weight_shape2, flat_params, offset)
        x = F.dropout(x, p=mlp.drop2.p, training=self.training)
        return x, offset

    def apply_attn(self, attn, x, flat_params, offset):
        B, N, C = x.shape

        # Extract qkv parameters
        qkv_weight_size = attn.qkv.weight.numel()
        qkv_bias_size = attn.qkv.bias.numel()
        qkv_weight = flat_params[offset:offset + qkv_weight_size].view_as(attn.qkv.weight)
        offset += qkv_weight_size
        qkv_bias = flat_params[offset:offset + qkv_bias_size].view_as(attn.qkv.bias)
        offset += qkv_bias_size

        # Compute qkv
        qkv = F.linear(x, qkv_weight, qkv_bias)
        qkv = qkv.reshape(B, N, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled Dot-Product Attention
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn_scores = attn_scores.softmax(dim=-1)
        attn_scores = F.dropout(attn_scores, p=attn.attn_drop.p, training=self.training)

        # Attention output
        x = (attn_scores @ v).transpose(1, 2).reshape(B, N, C)

        # Output projection
        proj_weight_size = attn.proj.weight.numel()
        proj_bias_size = attn.proj.bias.numel()
        proj_weight = flat_params[offset:offset + proj_weight_size].view_as(attn.proj.weight)
        offset += proj_weight_size
        proj_bias = flat_params[offset:offset + proj_bias_size].view_as(attn.proj.bias)
        offset += proj_bias_size

        x = F.linear(x, proj_weight, proj_bias)
        x = F.dropout(x, p=attn.proj_drop.p, training=self.training)
        return x, offset

    def num_flat_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, flat_params):
        offset = 0
        B = x.shape[0]

        # Extract cls_token from flat_params
        cls_token_size = self.cls_token.numel()
        cls_token = flat_params[offset:offset + cls_token_size].view_as(self.cls_token)
        offset += cls_token_size
        cls_token = cls_token.expand(B, -1, -1)  # Expand for batch size

        # Extract pos_embed from flat_params
        pos_embed_size = self.pos_embed.numel()
        pos_embed = flat_params[offset:offset + pos_embed_size].view_as(self.pos_embed)
        offset += pos_embed_size

        x = self.patch_embed(x)  # x has shape (B, N_patches, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # Concatenate cls_token

        x = x + pos_embed  # Add positional embeddings
        x = self.pos_drop(x)

        for blk in self.blocks:
            # LayerNorm 1
            x_norm, offset = self.apply_layernorm(blk.norm1, x, flat_params, offset)

            # Self-Attention
            x_attn, offset = self.apply_attn(blk.attn, x_norm, flat_params, offset)
            x = x + x_attn  # Residual connection

            # LayerNorm 2
            x_norm, offset = self.apply_layernorm(blk.norm2, x, flat_params, offset)

            # MLP
            x_mlp, offset = self.apply_mlp(blk.mlp, x_norm, flat_params, offset)
            x = x + x_mlp  # Residual connection

        # Final LayerNorm
        x, offset = self.apply_layernorm(self.norm, x, flat_params, offset)

        x = self.pre_logits(x)
        x = x[:, 0]  # Take the cls token

        # Head layer
        head_weight_size = self.head.weight.numel()
        head_bias_size = self.head.bias.numel()
        head_weight = flat_params[offset:offset + head_weight_size].view_as(self.head.weight)
        offset += head_weight_size
        head_bias = flat_params[offset:offset + head_bias_size].view_as(self.head.bias)
        offset += head_bias_size
        x = F.linear(x, head_weight, head_bias)

        return x

# # Test the VitNet class
# model = VitNet(num_classes=10)
# inputs = torch.randn(1, 3, 224, 224)

# # Calculating flat parameters size to mimic the parameters being passed as a flat tensor
# flat_params = torch.randn(model.num_flat_parameters())

# output = model(inputs, flat_params)
# print(output)
