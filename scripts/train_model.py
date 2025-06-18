import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm

from patchmatch.datasets import PatchTripletDataset
from patchmatch.models import PatchMatchTripletNetwork, TripletLoss


def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_dataloader(npz_path, batch_size, shuffle=True):
    dataset = PatchTripletDataset(npz_path)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_epoch(model, dataloader, loss_fn, device, epoch, optimizer=None, phase="Train"):
    is_train = phase == "Train"
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_samples = 0
    total_ap_dist = 0.0
    total_an_dist = 0.0

    progress_bar = tqdm(dataloader, desc=f"[Epoch {epoch}] {phase}", leave=False)

    with torch.set_grad_enabled(is_train):
        for step, (anchor, positive, negative) in enumerate(progress_bar):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            emb_anchor, emb_positive, emb_negative = model(anchor, positive, negative)
            loss = loss_fn(emb_anchor, emb_positive, emb_negative)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_samples += anchor.size(0)

            ap_dist = torch.norm(emb_anchor - emb_positive, dim=1).sum().item()
            an_dist = torch.norm(emb_anchor - emb_negative, dim=1).sum().item()
            total_ap_dist += ap_dist
            total_an_dist += an_dist

            progress_bar.set_postfix(loss=loss.item(), avg_loss=total_loss / (step + 1))

    avg_loss = total_loss / len(dataloader)
    avg_ap_dist = total_ap_dist / total_samples
    avg_an_dist = total_an_dist / total_samples

    print(f"[Epoch {epoch}] {phase} | Loss: {avg_loss:.4f} | "
          f"Avg Pos Dist: {avg_ap_dist:.4f} | Avg Neg Dist: {avg_an_dist:.4f}")

    return avg_loss


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[INFO] Using device: {device} ({device_name})")

    train_loader = prepare_dataloader(config["dataset"]["train"], config["training"]["batch_size"], shuffle=True)
    val_loader = prepare_dataloader(config["dataset"]["valid"], config["training"]["batch_size"], shuffle=False)

    model = PatchMatchTripletNetwork(embedding_dim=config["model"]["embedding_dim"]).to(device)
    loss_fn = TripletLoss(margin=config["model"].get("margin", 1.0))
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    log_dir = os.path.join(config["model_output_dir"], "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    best_model_path = os.path.join(config["model_output_dir"], "best_model.pt")
    final_model_path = os.path.join(config["model_output_dir"], "final_model.pt")

    best_val_loss = float("inf")

    for epoch in range(config["training"]["epochs"]):
        train_loss = evaluate_epoch(model, train_loader, loss_fn, device, epoch, optimizer=optimizer, phase="Train")
        val_loss = evaluate_epoch(model, val_loader, loss_fn, device, epoch, phase="Validation")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] Best model saved to {best_model_path}")

    torch.save(model.state_dict(), final_model_path)
    print(f"[INFO] Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
