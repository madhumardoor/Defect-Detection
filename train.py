# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 16:42:30 2025

@author: PC
"""

# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. Define the CNN model (same as in your app)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 2. Prepare dummy dataset (e.g., CIFAR-2 (Normal vs Defect))
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# For demo, use CIFAR-10 dataset and map some classes
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# Only take classes: airplane (0) and automobile (1) to simulate Normal/Defect
idx = (torch.tensor(trainset.targets) == 0) | (torch.tensor(trainset.targets) == 1)
trainset.targets = torch.tensor(trainset.targets)[idx]
trainset.data = trainset.data[idx.numpy()]

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

# 3. Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):  # 5 epochs for demo
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss:.3f}")

print("Training done ✅")

# 4. Save the trained model
torch.save(model.state_dict(), 'model.pth')
print("Saved model to model.pth ✅")
