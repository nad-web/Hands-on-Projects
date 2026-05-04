import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Standard Conv2d + BatchNorm + ReLU + MaxPool2d block."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    def forward(self, x):
        return self.block(x)

class MNIST_CNN(nn.Module):
    """CNN with configurable depth for concept emergence analysis.
    
    Args:
        depth: Number of convolutional blocks (2=Shallow, 3=Medium, 5=Deep)
    """
    def __init__(self, depth=3):
        super().__init__()
        self.depth = depth
        # Build convolutional blocks with doubling channels
        layers = []
        c = 1  # Input: 1 channel (grayscale)
        for i in range(depth):
            layers.append(ConvBlock(c, 16 * (2 ** i)))
            c = 16 * (2 ** i)
        self.conv_layers = nn.ModuleList(layers)
        # Compute flattened dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            for lyr in self.conv_layers:
                dummy = lyr(dummy)
            flat_dim = dummy.view(1, -1).shape[1]
        # Two-layer FC classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)  # 10 digit classes
        )

    def forward(self, x):
        for lyr in self.conv_layers:
            x = lyr(x)
        return self.classifier(x)

    def get_activations(self, x, layer_idx):
        """Extract global-average-pooled activations at specified layer.
        Returns: Tensor of shape (batch_size, channels)
        """
        out = x
        for i, lyr in enumerate(self.conv_layers):
            out = lyr(out)
            if i == layer_idx:
                return out.mean(dim=(2, 3))  # GAP: (B, C, H, W) -> (B, C)
        raise ValueError(f'layer_idx {layer_idx} out of range [0, {self.depth-1}]')
