import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
import numpy as np
from torchvision import transforms

from patchmatch.datasets.data_loader import create_dataloader
from patchmatch.models import PatchMatchTripletNetwork, TripletLoss
from utils.utils import evaluate_epoch, compute_recall_map1, save_plot, show_model_summary, mine_hard_negatives

def load_config(path="config/config.yaml"):
    """
    Load training configuration from a YAML file.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_augmentation_pipeline():
    """
    Define the online data augmentation pipeline.
    """
    return transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=0.5)
    ])

def train_with_online_mining(model, dataloader, loss_fn, device, optimizer, epoch):
    """
    Runs one epoch of training using online hard negative mining with augmented patches.
    """
    model.train()

    total_loss = 0.0
    total_samples = 0
    total_ap_dist = 0.0
    total_an_dist = 0.0
    triplet_correct = 0

    progress_bar = tqdm(dataloader, desc=f"[Epoch {epoch}] Training", leave=False)

    for step, (anchor, positive, _) in enumerate(progress_bar):
        anchor, positive = anchor.to(device), positive.to(device)

        emb_anchor = model.encoder(anchor)
        emb_positive = model.encoder(positive)

        emb_negative = mine_hard_negatives(emb_anchor, emb_positive)

        loss = loss_fn(emb_anchor, emb_positive, emb_negative)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_samples += anchor.size(0)

        ap_dist = torch.norm(emb_anchor - emb_positive, dim=1)
        an_dist = torch.norm(emb_anchor - emb_negative, dim=1)
        total_ap_dist += ap_dist.sum().item()
        total_an_dist += an_dist.sum().item()

        triplet_correct += torch.sum(ap_dist + loss_fn.margin < an_dist).item()

        progress_bar.set_postfix(loss=loss.item(), avg_loss=total_loss / (step + 1))

    avg_loss = total_loss / len(dataloader)
    avg_ap_dist = total_ap_dist / total_samples
    avg_an_dist = total_an_dist / total_samples
    pn_gap = avg_an_dist - avg_ap_dist
    triplet_acc = 100.0 * triplet_correct / total_samples

    print(f"[INFO][Epoch {epoch}] Train | Loss: {avg_loss:.4f} | "
          f"Avg Pos Dist: {avg_ap_dist:.4f} | Avg Neg Dist: {avg_an_dist:.4f} | "
          f"PN Gap: {pn_gap:.4f} | Triplet Acc: {triplet_acc:.2f}%")

    return avg_loss, avg_ap_dist, avg_an_dist, pn_gap, triplet_acc

def main():
    """
    Main training loop for PatchMatch Triplet Network using online hard negative mining with data augmentation.
    """
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[INFO] Using device: {device} ({device_name})")

    print("[INFO] Preparing train dataloader with augmentation...")
    augmentation = get_augmentation_pipeline()
    train_loader = create_dataloader(config["dataset"]["train"], config["training"]["batch_size"], shuffle=True, transform=augmentation)

    print("[INFO] Preparing validation dataloader...")
    val_loader = create_dataloader(config["dataset"]["valid"], config["training"]["batch_size"], shuffle=False)

    print("[INFO] Initializing model...")
    model = PatchMatchTripletNetwork(embedding_dim=config["model"]["embedding_dim"]).to(device)
    show_model_summary(model, input_shape=tuple(config["model"]["input_shape"][:2]), device=device)

    loss_fn = TripletLoss(margin=config["model"].get("margin", 1.0))
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    log_dir = os.path.join(config["model_output_dir"], "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    best_model_path = os.path.join(config["model_output_dir"], "best_model.pt")
    final_model_path = os.path.join(config["model_output_dir"], "final_model.pt")

    best_val_loss = float("inf")
    history = {
        "train_loss": [], "val_loss": [],
        "train_pn_gap": [], "val_pn_gap": [],
        "train_acc": [], "val_acc": [],
        "recall@1": [], "mAP@1": []
    }

    for epoch in range(config["training"]["epochs"]):
        train_loss, _, _, train_gap, train_acc = train_with_online_mining(
            model, train_loader, loss_fn, device, optimizer, epoch
        )

        val_loss, _, _, val_gap, val_acc = evaluate_epoch(
            model, val_loader, loss_fn, device, epoch, phase="Validation"
        )

        recall1, map1 = compute_recall_map1(model, config["dataset"]["valid"], device)
        print(f"[INFO][Epoch {epoch}] Validation | Recall@1: {recall1 * 100:.2f}% | mAP@1: {map1 * 100:.2f}%")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_pn_gap"].append(train_gap)
        history["val_pn_gap"].append(val_gap)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["recall@1"].append(recall1 * 100)
        history["mAP@1"].append(map1 * 100)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] Best model saved to {best_model_path}")

    torch.save(model.state_dict(), final_model_path)
    print(f"[INFO] Final model saved to {final_model_path}")

    plot_dir = os.path.join(log_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    save_plot({"Train": history["train_loss"], "Val": history["val_loss"]},
              ylabel="Triplet Loss", title="Triplet Loss over Epochs",
              save_path=os.path.join(plot_dir, "triplet_loss.png"))

    save_plot({"Train": history["train_pn_gap"], "Val": history["val_pn_gap"]},
              ylabel="Positive-Negative Gap", title="PN Gap over Epochs",
              save_path=os.path.join(plot_dir, "pn_gap.png"))

    save_plot({"Train": history["train_acc"], "Val": history["val_acc"]},
              ylabel="Triplet Accuracy (%)", title="Triplet Accuracy over Epochs",
              save_path=os.path.join(plot_dir, "triplet_accuracy.png"))

    save_plot({"Recall@1": history["recall@1"], "mAP@1": history["mAP@1"]},
              ylabel="Metric (%)", title="Recall@1 and mAP@1 over Epochs",
              save_path=os.path.join(plot_dir, "retrieval_metrics.png"))

if __name__ == "__main__":
    main()
