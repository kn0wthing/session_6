import torch
import torch.nn as nn

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        self.features = nn.Sequential(
            # First block layer - reduce initial channels
            # input size: 28x28x1
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # output: 28x28x8
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # nn.Dropout(0.1),  # Added earlier

            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # output: 28x28x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.Dropout(0.1),  # Added earlier
            
            # Transition layer
            nn.MaxPool2d(2, 2),  # output: 14x14x16
            nn.Dropout(0.1),   # Added now
            
            # Second block layer - efficient channel usage
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # output: 14x14x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # output: 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Transition layer
            nn.MaxPool2d(2, 2),  # output: 7x7x32
            nn.Dropout(0.1),   # Added now
            
            # Third block with channel reduction
            nn.Conv2d(32, 16, kernel_size=1),  # output: 7x7x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.Dropout(0.1),  # Added earlier
            
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # output: 7x7x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.Dropout(0.1),  # Added earlier
            
            # Final layers
            nn.MaxPool2d(2, 2),  # output: 3x3x16
            
            nn.Conv2d(16, 16, kernel_size=1),  # output: 3x3x16
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d(1),  # output: 1x1x16
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16, 10)  # input: 16, output: 10 (number of classes)
        )
        
    def forward(self, x):
        x = self.features(x)  # input: Nx1x28x28, output: Nx16x1x1
        x = x.view(-1, 16)    # flatten: Nx16
        x = self.classifier(x) # output: Nx10
        return x
