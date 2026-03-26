import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    CNN map encoder for binary occupancy maps.

    Architecture: 4 conv-pool blocks, then 2 FC layers.

    For image_size=64:
        (in_channels, 64, 64) -> pool x4 -> (128, 4, 4) -> flatten 2048 -> output_dim

    Args:
        image_size  : spatial size of input map (H == W). Must be divisible by 16.
        output_dim  : dimension of the output latent vector.
        in_channels : number of input channels (1 for binary map, 3 for map+start+goal blobs).
    """
    def __init__(self, image_size: int = 64, output_dim: int = 128, in_channels: int = 1):
        super().__init__()
        assert image_size % 16 == 0, "image_size must be divisible by 16 (4 pool layers)"

        self.conv1 = nn.Conv2d(in_channels, 16,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16,  32,  kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32,  64,  kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64,  128, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)

        # After 4 pool operations: image_size // 16
        final_size = image_size // 16          # e.g. 64//16 = 4
        flat_size  = 128 * final_size * final_size  # e.g. 128*4*4 = 2048

        self.fc1     = nn.Linear(flat_size, 256)
        self.fc2     = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) or (B, H, W) for single-channel
        Returns:
            latent: (B, output_dim)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)              # (B, 1, H, W)

        x = self.pool(F.relu(self.conv1(x)))  # (B, 16,  H/2,  W/2)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 32,  H/4,  W/4)
        x = self.pool(F.relu(self.conv3(x)))  # (B, 64,  H/8,  W/8)
        x = self.pool(F.relu(self.conv4(x)))  # (B, 128, H/16, W/16)

        x = x.view(x.size(0), -1)            # (B, flat_size)
        x = F.relu(self.fc1(x))              # (B, 256)
        x = self.dropout(x)
        x = self.fc2(x)                      # (B, output_dim)

        return x