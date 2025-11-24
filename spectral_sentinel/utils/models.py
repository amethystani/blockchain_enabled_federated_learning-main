"""
Model architectures for federated learning experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN for MNIST/CIFAR-10.
    
    Architecture:
    - Conv1: 3→32
    - Conv2: 32→64
    - FC1: 64*7*7 → 128
    - FC2: 128 → num_classes
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 1):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size based on input
        # For MNIST (28x28): after 2 pools → 7x7
        # For CIFAR (32x32): after 2 pools → 8x8
        self.fc_input_size = 64 * 8 * 8 if input_channels == 3 else 64 * 7 * 7
        
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        # Conv layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class LeNet5(nn.Module):
    """LeNet-5 architecture for MNIST."""
    
    def __init__(self, num_classes: int = 10):
        super(LeNet5, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        self.pool = nn.AvgPool2d(2, 2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class ResNetBlock(nn.Module):
    """Basic ResNet block."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    """ResNet-18 for CIFAR-10/100."""
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super(ResNet18, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class ResNet50(nn.Module):
    """ResNet-50 for Phase 3A: Medium-Scale (~25M params)."""
    
    def __init__(self, num_classes: int = 62, input_channels: int = 1):
        super(ResNet50, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7,
                              stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet-50 uses Bottleneck blocks: [3, 4, 6, 3]
        self.layer1 = self._make_bottleneck_layer(64, 64, 256, 3, stride=1)
        self.layer2 = self._make_bottleneck_layer(256, 128, 512, 4, stride=2)
        self.layer3 = self._make_bottleneck_layer(512, 256, 1024, 6, stride=2)
        self.layer4 = self._make_bottleneck_layer(1024, 512, 2048, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
    
    def _make_bottleneck_layer(self, in_channels, mid_channels, out_channels, 
                               num_blocks, stride):
        layers = []
        # First block with stride
        layers.append(BottleneckBlock(in_channels, mid_channels, out_channels, stride))
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(BottleneckBlock(out_channels, mid_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class BottleneckBlock(nn.Module):
    """Bottleneck block for ResNet-50."""
    
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ViTSmall(nn.Module):
    """Vision Transformer Small for Phase 3B (~22M params)."""
    
    def __init__(self, num_classes: int = 200, img_size: int = 64, 
                 patch_size: int = 8, embed_dim: int = 384, depth: int = 12,
                 num_heads: int = 6, mlp_ratio: float = 4.0):
        super(ViTSmall, self).__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, 
                                     stride=patch_size)
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # B, embed_dim, H/P, W/P
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Classification head (use class token)
        return self.head(x[:, 0])


class TransformerBlock(nn.Module):
    """Transformer block for ViT."""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super(TransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim)
        )
    
    def forward(self, x):
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class GPT2Medium(nn.Module):
    """GPT-2 Medium wrapper for Phase 3C (~345M params)."""
    
    def __init__(self, vocab_size: int = 50257, num_classes: int = None):
        super(GPT2Medium, self).__init__()
        
        # Note: This is a placeholder. Actual implementation should use
        # transformers library: GPT2LMHeadModel.from_pretrained('gpt2-medium')
        # For now, we create a minimal version
        
        self.embed_dim = 1024
        self.num_heads = 16
        self.num_layers = 24
        
        self.token_embed = nn.Embedding(vocab_size, self.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, self.embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads)
            for _ in range(self.num_layers)
        ])
        
        self.norm = nn.LayerNorm(self.embed_dim)
        self.lm_head = nn.Linear(self.embed_dim, vocab_size, bias=False)
    
    def forward(self, input_ids, attention_mask=None):
        seq_len = input_ids.shape[1]
        
        # Token + position embeddings
        x = self.token_embed(input_ids)
        x = x + self.pos_embed[:, :seq_len, :]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits


def get_model(model_name: str, num_classes: int = 10, input_channels: int = 1) -> nn.Module:
    """
    Get model by name.
    
    Args:
        model_name: Model name - 'simple_cnn', 'lenet5', 'resnet18', 'resnet50',
                    'vit_small', 'gpt2_medium'
        num_classes: Number of output classes
        input_channels: Number of input channels
        
    Returns:
        Model instance
    """
    model_name = model_name.lower()
    
    if model_name == 'simple_cnn':
        return SimpleCNN(num_classes, input_channels)
    elif model_name == 'lenet5':
        return LeNet5(num_classes)
    elif model_name == 'resnet18':
        return ResNet18(num_classes, input_channels)
    elif model_name == 'resnet50':
        return ResNet50(num_classes, input_channels)
    elif model_name == 'vit_small':
        return ViTSmall(num_classes)
    elif model_name == 'gpt2_medium':
        return GPT2Medium()
    else:
        raise ValueError(f"Unknown model: {model_name}")
