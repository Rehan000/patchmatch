import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from patchmatch.datasets import PatchPairDataset
from patchmatch.models import PatchMatchSiameseDescriptor, ContrastiveLoss

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def prepare_dataloader(npz_path, batch_size, shuffle=True):
    """
    Prepares a PyTorch DataLoader from the given NPZ file.
    """
    dataset = PatchPairDataset(npz_path)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    for step, ((p1, p2), labels) in enumerate(dataloader):
        p1, p2, labels = p1.to(device), p2.to(device), labels.to(device)

        optimizer.zero_grad()
        distances = model(p1, p2)
        loss = loss_fn(distances, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if step % 100 == 0:
            print(f"[Epoch {epoch} Step {step}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate(model, dataloader, loss_fn, device, epoch):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for (p1, p2), labels in dataloader:
            p1, p2, labels = p1.to(device), p2.to(device), labels.to(device)
            distances = model(p1, p2)
            loss = loss_fn(distances, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def main():
    config = load_config()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader = prepare_dataloader(config["dataset"]["train"], config["training"]["batch_size"], shuffle=True)
    val_loader = prepare_dataloader(config["dataset"]["valid"], config["training"]["batch_size"], shuffle=False)

    # Model
    model = PatchMatchSiameseDescriptor(
        embedding_dim=config["model"]["embedding_dim"]
    ).to(device)

    # Loss and Optimizer
    loss_fn = ContrastiveLoss(margin=config["model"]["margin"])
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # Output directory
    log_dir = os.path.join(config["model_output_dir"], "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    best_val_loss = float("inf")
    best_model_path = os.path.join(config["model_output_dir"], "best_model.pt")

    # Training loop
    for epoch in range(config["training"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        val_loss = validate(model, val_loader, loss_fn, device, epoch)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] Best model saved to {best_model_path}")

    # Save final model
    final_model_path = os.path.join(config["model_output_dir"], "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"[INFO] Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()
