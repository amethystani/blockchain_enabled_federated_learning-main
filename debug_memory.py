import torch
import sys
import os

print(f"PID: {os.getpid()}")
print("Imported torch")

try:
    x = torch.randn(1000, 1000)
    print("Created 1000x1000 tensor")
    
    import torchvision
    from torchvision import datasets, transforms
    print("Imported torchvision")
    
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    print(f"Loaded MNIST: {len(dataset)} samples")
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    print("Created DataLoader")
    
    for batch in loader:
        print(f"Loaded batch: {batch[0].shape}")
        break
        
except Exception as e:
    print(f"Error: {e}")
