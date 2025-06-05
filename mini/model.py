import torch
import torch.nn as nn

class AODModel(nn.Module):
    def __init__(self):
        super(AODModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(13, 32, kernel_size=3, padding=1),  # conv.0
            nn.ReLU(),                                    # conv.1
            nn.MaxPool2d(2),                              # conv.2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # conv.3
            nn.ReLU(),                                    # conv.4
            nn.AdaptiveAvgPool2d((4, 4)),                 # conv.5
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                                 # fc.0
            nn.Linear(64 * 4 * 4, 128),                   # fc.1
            nn.ReLU(),                                    # fc.2
            nn.Linear(128, 1)                             # fc.3
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x