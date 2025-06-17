import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm

from patchmatch.datasets import PatchPairDataset
from patchmatch.models import PatchMatchSiameseDescriptor, ContrastiveLoss


def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_dataloader(npz_path, batch_size, shuffle=True):
    dataset = PatchPairDataset(npz_path)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_epoch(model, dataloader, loss_fn, device, epoch, phase="Train", margin=1.0, threshold=0.5):
    model.train() if phase == "Train" else model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    margin_violations = 0
    pos_dist_sum = 0
    neg_dist_sum = 0
    pos_count = 0
    neg_count = 0

    progress_bar = tqdm(dataloader, desc=f"[Epoch {epoch}] {phase}", leave=False)
    with torch.set_grad_enabled(phase == "Train"):
        for step, ((p1, p2), labels) in enumerate(progress_bar):
            p1, p2, labels = p1.to(device), p2.to(device), labels.to(device)

            distances = model(p1, p2)
            loss = loss_fn(distances, labels)

            if phase == "Train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds = (distances < threshold).float()
            total_correct += (preds.squeeze() == labels).sum().item()
            total_samples += labels.size(0)

            # Mean distances by label
            pos_mask = labels == 1
            neg_mask = labels == 0
            if pos_mask.any():
                pos_dist_sum += distances[pos_mask].sum().item()
                pos_count += pos_mask.sum().item()
            if neg_mask.any():
                neg_dist_sum += distances[neg_mask].sum().item()
                neg_count += neg_mask.sum().item()
                margin_violations += (distances[neg_mask] < margin).sum().item()

            avg_loss = total_loss / (step + 1)
            progress_bar.set_postfix(loss=loss.item(), avg=avg_loss)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    avg_pos_dist = pos_dist_sum / pos_count if pos_count > 0 else 0.0
    avg_neg_dist = neg_dist_sum / neg_count if neg_count > 0 else 0.0
    violation_rate = margin_violations / neg_count if neg_count > 0 else 0.0

    print(f"[Epoch {epoch}] {phase} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | "
          f"PosDist: {avg_pos_dist:.4f} | NegDist: {avg_neg_dist:.4f} | "
          f"Margin Violations: {violation_rate:.2%}")

    return avg_loss


def main():
    config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[INFO] Using device: {device} ({device_name})")

    train_loader = prepare_dataloader(
        config["dataset"]["train"],
        config["training"]["batch_size"],
        shuffle=True
    )
    val_loader = prepare_dataloader(
        config["dataset"]["valid"],
        config["training"]["batch_size"],
        shuffle=False
    )

    model = PatchMatchSiameseDescriptor(
        embedding_dim=config["model"]["embedding_dim"]
    ).to(device)

    global optimizer  # Needed for `evaluate_epoch`
    loss_fn = ContrastiveLoss(margin=config["model"]["margin"])
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    log_dir = os.path.join(
        config["model_output_dir"], "logs", datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    os.makedirs(log_dir, exist_ok=True)
    best_model_path = os.path.join(config["model_output_dir"], "best_model.pt")
    final_model_path = os.path.join(config["model_output_dir"], "final_model.pt")
    best_val_loss = float("inf")

    margin = config["model"].get("margin", 1.0)
    threshold = config["model"].get("decision_threshold", 0.5)

    for epoch in range(config["training"]["epochs"]):
        train_loss = evaluate_epoch(model, train_loader, loss_fn, device, epoch,
                                    phase="Train", margin=margin, threshold=threshold)
        val_loss = evaluate_epoch(model, val_loader, loss_fn, device, epoch,
                                  phase="Validation", margin=margin, threshold=threshold)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] Best model saved to {best_model_path}")

    torch.save(model.state_dict(), final_model_path)
    print(f"[INFO] Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
