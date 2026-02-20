import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Input: (batch_size, 1, 8, 8)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # (16, 8, 8)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # (32, 8, 8)
        self.pool = nn.MaxPool2d(2, 2)  # (32, 4, 4) after pooling
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (64, 4, 4)
        
        # Flatten: 64 * 2 * 2 = 256
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x shape: (batch_size, 8, 8) -> add channel dimension
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # (batch_size, 1, 8, 8)
        
        x = F.relu(self.conv1(x))  # (batch_size, 16, 8, 8)
        x = F.relu(self.conv2(x))  # (batch_size, 32, 8, 8)
        x = self.pool(x)           # (batch_size, 32, 4, 4)
        x = F.relu(self.conv3(x))  # (batch_size, 64, 4, 4)
        x = self.pool(x)           # (batch_size, 64, 2, 2)
        
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 256)
        x = F.relu(self.fc1(x))    # (batch_size, 128)
        x = self.dropout(x)
        x = self.fc2(x)            # (batch_size, 64)
        
        return x