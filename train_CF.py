import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from train_Sentinel_loadonthego import HSIDataset, get_file_lists
from cfmodel import chromaformer_t, chromaformer_s, chromaformer_b, chromaformer_l, chromaformer_h

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_chromaformer(model, train_loader, val_loader, epochs=10, learning_rate=1e-4, save_path="./checkpoints"):
    """Train ChromaFormer model locally without Accelerate."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            # Convert labels from one-hot to class indices
            if labels.ndim > 1:
                labels = torch.argmax(labels, dim=1)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze().long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                if labels.ndim > 1:
                    labels = torch.argmax(labels, dim=1)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels.long()).sum().item()

        acc = 100 * correct / total
        print(f"Validation Accuracy: {acc:.2f}%")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, "best_chromaformer.pth"))
            print("Saved best model!")

if __name__ == "__main__":
    # Data loading
    data_dir = r"F:\work\ieeetgrs\for_chromaformer\16_16"
    train_files, val_files = get_file_lists(data_dir, 0.8, 1.0)
    train_loader = DataLoader(HSIDataset(train_files), batch_size=16)
    val_loader = DataLoader(HSIDataset(val_files), batch_size=16)

    
    model = chromaformer_t(input_resolution=(16, 16), 
                       in_channels=11, 
                       window_size=4, 
                       num_classes=15)  

    
    train_chromaformer(model, train_loader, val_loader, epochs=20)
