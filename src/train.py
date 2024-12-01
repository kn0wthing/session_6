import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import MNISTNet
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def train_model():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomRotation(degrees=5),  # Added augmentation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))  # Added augmentation
    ])
    
    # Test transform without augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    full_train_dataset = datasets.MNIST(root='./data', 
                                      train=True,
                                      transform=transform,
                                      download=True)
    
    # Split into train and validation
    train_size = 50000
    val_size = 10000
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Override validation transform
    val_dataset.dataset.transform = test_transform
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    
    # Initialize model, loss, and optimizer
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)  # Increased learning rate
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                                            max_lr=0.01,
                                            epochs=20,
                                            steps_per_epoch=len(train_loader))
    
    # Training loop
    best_acc = 0.0
    epochs = 20
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            
        # Print average training loss
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.4f}')
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')
        
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            
        if accuracy > 99.4:
            print(f"Reached target accuracy of 99.4% at epoch {epoch+1}")
            break
    
    print(f'Best validation accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    train_model() 