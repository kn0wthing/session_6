import torch
import torch.nn as nn

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        self.features = nn.Sequential(
            # Initial conv layer
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # 1x1 conv for channel attention
            nn.Conv2d(16, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            
            # Second conv block
            nn.Conv2d(8, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            
            # 1x1 conv for channel reduction
            nn.Conv2d(24, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            
            # Final conv layer
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(32, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 32)
        x = self.classifier(x)
        return x 