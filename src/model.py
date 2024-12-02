import torch
import torch.nn as nn

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        self.features = nn.Sequential(
            # First block layer - reduce initial channels
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Transition layer
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),
            
            # Second block layer - efficient channel usage
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Transition layer
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),
            
            # Third block with channel reduction
            nn.Conv2d(32, 16, kernel_size=1),  # Channel reduction
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Final layers
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),
            
            nn.Conv2d(32, 16, kernel_size=1),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16)
        x = self.classifier(x)
        return x
