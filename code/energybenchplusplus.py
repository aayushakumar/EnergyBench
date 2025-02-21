import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from codecarbon import EmissionsTracker
import time
import numpy as np
import pandas as pd
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Hyperparameters
batch_sizes = [8, 16, 32]
num_epochs = 1
num_inference_batches = 100
power_limit = 50  # Watts, simulate edge constraint (optional; requires nvidia-smi)

# Data loading
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # For STL-10 compatibility
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

datasets = {
    "CIFAR-10": torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform),
    "STL-10": torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
}
test_datasets = {
    "CIFAR-10": torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform),
    "STL-10": torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)
}

# Models
models = {
    "MobileNetV2": torchvision.models.mobilenet_v2(pretrained=True),
    "ResNet-18": torchvision.models.resnet18(pretrained=True),
    "EfficientNet-B0": torchvision.models.efficientnet_b0(pretrained=True),
    "SqueezeNet": torchvision.models.squenzenet1_0(pretrained=True),
    "TinyBERT": None  # Placeholder; replace with vision model if needed
}

# Training function
def train_model(model, loader, epochs, tracker):
    model.train()
    tracker.start()
    start_time = time.time()
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print(f"Step [{i}/{len(loader)}], Loss: {loss.item():.4f}")
    training_time = time.time() - start_time
    emissions = tracker.stop()
    return training_time, emissions

# Inference function
def inference_model(model, loader, num_batches, tracker):
    model.eval()
    tracker.start()
    start_time = time.time()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if i >= num_batches:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    inference_time = time.time() - start_time
    emissions = tracker.stop()
    accuracy = 100 * correct / total
    return inference_time, emissions, accuracy

# Main experiment
results = []
if not os.path.exists("./results"):
    os.makedirs("./results")

# Optional: Set power limit (run in terminal first: nvidia-smi -pl 50)
print("Set GPU power limit to 50W manually via 'nvidia-smi -pl 50' if desired.")

for model_name, model in models.items():
    if model is None:  # Skip TinyBERT or replace
        print("Skipping TinyBERT; replace with a vision model if needed.")
        continue
    model = model.to(device)
    for dataset_name, dataset in datasets.items():
        for bs in batch_sizes:
            print(f"\nRunning {model_name} on {dataset_name}, Batch Size: {bs}")
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_datasets[dataset_name], batch_size=bs, shuffle=False)

            # Training
            tracker = EmissionsTracker(project_name=f"{model_name}_{dataset_name}_train_bs{bs}", output_dir="./results")
            train_time, train_emissions = train_model(model, train_loader, num_epochs, tracker)

            # Inference
            tracker = EmissionsTracker(project_name=f"{model_name}_{dataset_name}_infer_bs{bs}", output_dir="./results")
            infer_time, infer_emissions, accuracy = inference_model(model, test_loader, num_inference_batches)

            # D-EAER (Dynamic EAER: Accuracy / (Energy * Runtime Factor))
            runtime_factor = bs / 16  # Normalize by baseline batch size
            d_eaer = accuracy / (infer_emissions * runtime_factor)

            # Store results
            result = {
                "model": model_name,
                "dataset": dataset_name,
                "batch_size": bs,
                "train_time": train_time,
                "train_energy_kWh": train_emissions,
                "infer_time": infer_time,
                "infer_energy_kWh": infer_emissions,
                "accuracy": accuracy,
                "d_eaer": d_eaer
            }
            results.append(result)
            print(f"{model_name} {dataset_name} BS{bs} - D-EAER: {d_eaer:.2f}, Accuracy: {accuracy:.2f}%")

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("./results/energybenchplusplus_results.csv", index=False)

# Basic analysis plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
for model_name in df["model"].unique():
    subset = df[df["model"] == model_name]
    plt.scatter(subset["infer_energy_kWh"], subset["accuracy"], label=model_name)
plt.xlabel("Inference Energy (kWh)")
plt.ylabel("Accuracy (%)")
plt.title("Energy vs Accuracy Across Models")
plt.legend()
plt.savefig("./results/energy_vs_accuracy.png")
plt.show()
