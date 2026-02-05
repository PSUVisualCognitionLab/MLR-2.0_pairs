import torch
import torch.nn as nn

class SmallTwoHeadCNN(nn.Module):
    """
    One shared CNN trunk for the concatenated bitmap (1×28×56),
    two classifier heads: left and right.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # -> (64, 14, 28)
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # -> (128, 7, 14)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.head_left = nn.Linear(512, num_classes)
        self.head_right = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        h = self.flatten(h)
        h = self.fc(h)
        return self.head_left(h), self.head_right(h)
