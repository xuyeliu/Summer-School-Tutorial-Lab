"""
Generate pre-trained checkpoints for the Privacy Lab Tutorial.
Trains models at multiple epsilon values for the privacy-utility trade-off visualization.

Usage: python generate_checkpoints.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
from opacus import PrivacyEngine

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(42)
np.random.seed(42)

os.makedirs("checkpoints", exist_ok=True)

# Data
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

info = INFO['pneumoniamnist']
DataClass = getattr(medmnist, info['python_class'])
train_dataset = DataClass(split='train', transform=data_transform, download=True)
test_dataset = DataClass(split='test', transform=data_transform, download=True)

# Split training data: 40% members, 40% non-members, 20% validation
total_size = len(train_dataset)
indices = np.arange(total_size)
np.random.shuffle(indices)
split1 = int(0.4 * total_size)
split2 = int(0.8 * total_size)

member_indices = indices[:split1]
nonmember_indices = indices[split1:split2]

member_dataset = Subset(train_dataset, member_indices)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Models
class SimpleFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def create_model(model_type):
    if model_type == "cnn":
        return SimpleCNN().to(device)
    else:
        return SimpleFC().to(device)


def evaluate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def train_no_dp(model_type, epochs=30):
    """Train without differential privacy."""
    model = create_model(model_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(member_dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    acc = evaluate_accuracy(model, test_loader)
    print(f"  No-DP model test accuracy: {acc:.1f}%")
    return model


def train_with_dp(model_type, epsilon, epochs=20):
    """Train with DP-SGD at given epsilon."""
    model = create_model(model_type)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(member_dataset, batch_size=64, shuffle=True)

    DELTA = 1e-5
    MAX_GRAD_NORM = 1.0

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        target_epsilon=epsilon,
        target_delta=DELTA,
        max_grad_norm=MAX_GRAD_NORM,
    )

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Get the underlying model state dict
    model_state = model._module.state_dict() if hasattr(model, '_module') else model.state_dict()

    # Create a clean model and load state
    clean_model = create_model(model_type)
    clean_model.load_state_dict(model_state)

    acc = evaluate_accuracy(clean_model, test_loader)
    eps_spent = privacy_engine.get_epsilon(DELTA)
    print(f"  epsilon={epsilon}: test accuracy={acc:.1f}%, epsilon_spent={eps_spent:.2f}")
    return clean_model


# Generate checkpoints for both model types
epsilons = [0.5, 1.0, 2.0, 5.0, 10.0]

for model_type in ["fc", "cnn"]:
    print(f"\n{'='*50}")
    print(f"Training {model_type.upper()} models")
    print(f"{'='*50}")

    # No-DP model
    print(f"\nTraining without DP (epsilon=inf)...")
    model_inf = train_no_dp(model_type, epochs=30 if model_type == "cnn" else 25)
    torch.save(model_inf.state_dict(), f"checkpoints/{model_type}_eps_inf.pt")

    # DP models at various epsilon
    for eps in epsilons:
        print(f"\nTraining with epsilon={eps}...")
        model_dp = train_with_dp(model_type, eps, epochs=20)
        torch.save(model_dp.state_dict(), f"checkpoints/{model_type}_eps_{eps}.pt")

print(f"\n{'='*50}")
print("All checkpoints generated successfully!")
print(f"Files saved in: {os.path.abspath('checkpoints/')}")
print(f"{'='*50}")
