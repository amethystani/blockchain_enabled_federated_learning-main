import torch
import torch.nn as nn
import torch.optim as optim
from spectral_sentinel.utils.models import SimpleCNN
from spectral_sentinel.federated.data_loader import load_federated_data, create_data_loader
import sys
import os

print(f"PID: {os.getpid()}")

# Load data
print("Loading data...")
client_datasets, test_dataset = load_federated_data('mnist', num_clients=2)
train_loader = create_data_loader(client_datasets[0], batch_size=16)

# Create model
print("Creating model...")
model = SimpleCNN(num_classes=10, input_channels=1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Train
print("Starting training...")
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    if batch_idx % 10 == 0:
        print(f"Batch {batch_idx}: Loss {loss.item():.4f}")
    
    if batch_idx > 50:
        break

print("Training finished successfully")
