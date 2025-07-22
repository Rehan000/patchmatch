import os
import torch
import matplotlib.pyplot as plt
from utils.utils import save_smooth_plot


def load_training_history(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint['history']


def main():
    checkpoint_path = "model_outputs/checkpoints/epoch_49.pt"
    output_dir = "outputs/training_plots"
    os.makedirs(output_dir, exist_ok=True)

    history = load_training_history(checkpoint_path)

    # Plot Triplet Loss
    save_smooth_plot(
        {"Train": history["train_loss"]},
        ylabel="Triplet Loss",
        title="Triplet Loss over Epochs",
        save_path=os.path.join(output_dir, "train_triplet_loss.png")
    )

    # Plot Triplet Accuracy
    save_smooth_plot(
        {"Train": history["train_acc"]},
        ylabel="Triplet Accuracy (%)",
        title="Triplet Accuracy over Epochs",
        save_path=os.path.join(output_dir, "train_triplet_accuracy.png")
    )

    # Plot PN Gap
    save_smooth_plot(
        {"Train": history["train_pn_gap"]},
        ylabel="Positive-Negative Gap",
        title="PN Gap over Epochs",
        save_path=os.path.join(output_dir, "train_pn_gap.png")
    )

    # Plot Learning Rate
    save_smooth_plot(
        {"Train": history["learning_rate"]},
        ylabel="Learning Rate",
        title="Learning Rate Schedule",
        save_path=os.path.join(output_dir, "learning_rate.png")
    )

    print(f"[INFO] Training plots saved to {output_dir}")


if __name__ == "__main__":
    main()
