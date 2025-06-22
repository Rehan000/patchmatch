import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from patchmatch.datasets.data_loader import create_dataloader
from patchmatch.models import PatchMatchTripletNetwork, TripletLoss
from utils.utils import evaluate_epoch, compute_recall_map1, save_plot, show_model_summary

def load_config(path="config/config.yaml"):
    """
    Load training configuration from a YAML file.

    Args:
        path (str): Path to the YAML config file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    """
    Main training loop for PatchMatch Triplet Network.

    Loads configuration, initializes dataloaders, model, optimizer, and loss function,
    then runs training and validation for the specified number of epochs, saving the
    best and final model weights along with performance plots.
    """
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[INFO] Using device: {device} ({device_name})")

    print("[INFO] Preparing train dataloader...")
    train_loader = create_dataloader(config["dataset"]["train"], config["training"]["batch_size"], shuffle=True)
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
        train_loss, _, _, train_gap, train_acc = evaluate_epoch(
            model, train_loader, loss_fn, device, epoch, optimizer=optimizer, phase="Train")

        val_loss, _, _, val_gap, val_acc = evaluate_epoch(
            model, val_loader, loss_fn, device, epoch, phase="Validation")

        recall1, map1 = compute_recall_map1(model, config["dataset"]["valid"], device)
        print(f"[INFO][Epoch {epoch}] Validation | Recall@1: {recall1*100:.2f}% | mAP@1: {map1*100:.2f}%")

        # Save metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_pn_gap"].append(train_gap)
        history["val_pn_gap"].append(val_gap)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["recall@1"].append(recall1 * 100)
        history["mAP@1"].append(map1 * 100)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] Best model saved to {best_model_path}")

    # Save final model
    torch.save(model.state_dict(), final_model_path)
    print(f"[INFO] Final model saved to {final_model_path}")

    # Generate plots
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
