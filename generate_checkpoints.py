"""
Generate pre-trained checkpoints for the Privacy Lab Tutorial.
Trains models at multiple epsilon values for the privacy-utility trade-off visualization.

Every checkpoint uses ONE matched recipe (SGD, lr=0.1, batch 64, 60 epochs) so that
epsilon is the only variable along the sweep. The eps=inf checkpoint runs the same
recipe with clipping and noise switched off, and the control checkpoint runs it on the
leftover 20% of the training pool, which contains neither members nor non-members.

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

# Matched training recipe. EPOCHS must be large enough that a non-private model can
# actually memorize the member split, otherwise the noise level has nothing to suppress
# and the privacy-utility trade-off is invisible.
EPOCHS = 60
LR = 0.1
BATCH_SIZE = 64
MAX_GRAD_NORM = 1.0
DELTA = 1e-5
EPSILONS = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 200.0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(42)
np.random.seed(42)

os.makedirs("checkpoints", exist_ok=True)

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

info = INFO['pneumoniamnist']
DataClass = getattr(medmnist, info['python_class'])
train_dataset = DataClass(split='train', transform=data_transform, download=True)
test_dataset = DataClass(split='test', transform=data_transform, download=True)

# Split training data: 40% members, 40% non-members, leftover 20% for the control.
# This must stay identical to the notebook's member / non-member cut so the
# checkpoints and the notebook's member_loader / nonmember_loader refer to the
# same samples. The notebook does not load the leftover 20%.
total_size = len(train_dataset)
indices = np.arange(total_size)
np.random.shuffle(indices)
split1 = int(0.4 * total_size)
split2 = int(0.8 * total_size)

member_indices = indices[:split1]
nonmember_indices = indices[split1:split2]
control_indices = indices[split2:]

member_dataset = Subset(train_dataset, member_indices)
control_dataset = Subset(train_dataset, control_indices)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


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


def create_model():
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


def train_no_dp(dataset, epochs=EPOCHS):
    """Matched recipe with clipping and noise switched off."""
    torch.manual_seed(42)
    model = create_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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
    member_acc = evaluate_accuracy(model, DataLoader(dataset, batch_size=64))
    print(f"  test accuracy={acc:.1f}%, train accuracy={member_acc:.1f}%")
    return model


def train_with_dp(epsilon, epochs=EPOCHS):
    """Matched recipe with DP-SGD at the given epsilon."""
    torch.manual_seed(42)
    model = create_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(member_dataset, batch_size=BATCH_SIZE, shuffle=True)

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
            if images.size(0) == 0:      # Poisson sampling can yield an empty batch
                continue
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model_state = model._module.state_dict() if hasattr(model, '_module') else model.state_dict()
    clean_model = create_model()
    clean_model.load_state_dict(model_state)

    acc = evaluate_accuracy(clean_model, test_loader)
    member_acc = evaluate_accuracy(clean_model, DataLoader(member_dataset, batch_size=64))
    eps_spent = privacy_engine.get_epsilon(DELTA)
    print(f"  epsilon={epsilon}: test accuracy={acc:.1f}%, train accuracy={member_acc:.1f}%, "
          f"sigma={optimizer.noise_multiplier:.3f}, epsilon_spent={eps_spent:.2f}")
    return clean_model


print(f"\n{'='*60}")
print(f"Matched recipe: SGD lr={LR}, batch={BATCH_SIZE}, {EPOCHS} epochs, C={MAX_GRAD_NORM}")
print(f"Members: {len(member_dataset)}, non-members: {len(nonmember_indices)}, "
      f"control holdout: {len(control_dataset)}")
print(f"{'='*60}")

# Control: trained on the leftover holdout only, so it has seen neither the member nor
# the non-member samples. Whatever membership signal it shows is pure split noise, which
# gives Phase 4 an empirical "no leakage" reference to compare every DP model against.
print("\nTraining CONTROL model on the leftover holdout (no members, no non-members)...")
model_control = train_no_dp(control_dataset)
torch.save(model_control.state_dict(), "checkpoints/fc_control.pt")

# Non-private reference: same optimizer and epoch budget, no clipping and no noise.
print("\nTraining without DP (epsilon=inf)...")
model_inf = train_no_dp(member_dataset)
torch.save(model_inf.state_dict(), "checkpoints/fc_eps_inf.pt")

for eps in EPSILONS:
    print(f"\nTraining with epsilon={eps}...")
    model_dp = train_with_dp(eps)
    torch.save(model_dp.state_dict(), f"checkpoints/fc_eps_{eps}.pt")

print(f"\n{'='*60}")
print("All checkpoints generated successfully!")
print(f"Files saved in: {os.path.abspath('checkpoints/')}")
print(f"{'='*60}")
