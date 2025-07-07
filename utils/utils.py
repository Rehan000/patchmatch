import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from torch.nn.functional import normalize
from torchsummary import summary

def get_learning_rate(optimizer):
    """
    Retrieves the current learning rate from the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer in use.

    Returns:
        float: Current learning rate.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def evaluate_epoch(model, dataloader, loss_fn, device, epoch, optimizer=None, phase="Train"):
    """
    Runs one epoch of training or validation using pre-generated triplets (for validation).

    Args:
        model (nn.Module): The triplet network model.
        dataloader (DataLoader): Dataloader for the current phase.
        loss_fn (nn.Module): Triplet loss function.
        device (torch.device): Device to run the computations on.
        epoch (int): Current epoch number.
        optimizer (Optimizer, optional): Optimizer for training. Required if phase is 'Train'.
        phase (str): Either 'Train' or 'Validation'.

    Returns:
        Tuple[float, float, float, float, float]:
            - Average triplet loss
            - Average anchor-positive distance
            - Average anchor-negative distance
            - Positive-Negative distance gap
            - Triplet accuracy in percentage
    """
    is_train = phase == "Train"
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_samples = 0
    total_ap_dist = 0.0
    total_an_dist = 0.0
    triplet_correct = 0

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

    print(f"[INFO][Epoch {epoch}] {phase} | Loss: {avg_loss:.4f} | "
          f"Avg Pos Dist: {avg_ap_dist:.4f} | Avg Neg Dist: {avg_an_dist:.4f} | "
          f"PN Gap: {pn_gap:.4f} | Triplet Acc: {triplet_acc:.2f}%")

    return avg_loss, avg_ap_dist, avg_an_dist, pn_gap, triplet_acc

def mine_hard_negatives(emb_anchors, emb_positives):
    """
    Online Hard Negative Mining.

    Selects the hardest negative for each anchor within the batch by choosing the
    closest positive from other anchors' positives in embedding space.

    Args:
        emb_anchors (Tensor): Anchor embeddings, shape (B, D)
        emb_positives (Tensor): Positive embeddings, shape (B, D)

    Returns:
        Tensor: Hard negatives for each anchor, shape (B, D)
    """
    batch_size = emb_anchors.size(0)

    # Compute pairwise distances between anchors and positives
    dists = torch.cdist(emb_anchors, emb_positives, p=2)  # Shape: (B, B)

    # Mask the diagonal (positive pairs) by adding a large constant
    mask = torch.eye(batch_size, device=emb_anchors.device)
    dists = dists + mask * 1e5  # Ignore positive pairs by setting distance to a large value

    # Find the closest (hardest) negative for each anchor
    hard_neg_indices = torch.argmin(dists, dim=1)
    hard_negatives = emb_positives[hard_neg_indices]

    return hard_negatives

def compute_recall_map_k(model, npz_path, device, ks=[1, 3, 5]):
    """
    Computes Recall@k and mAP@k retrieval metrics for multiple k values using the validation dataset.

    Args:
        model (nn.Module): The trained triplet network.
        npz_path (str): Path to the validation dataset (.npz file).
        device (torch.device): Device to run computations on.
        ks (List[int]): List of k values for which to compute Recall@k and mAP@k.

    Returns:
        dict: Recall@k and mAP@k for all requested ks.
    """
    data = np.load(npz_path)
    triplets = data['triplets']

    anchors = triplets[..., 0]
    positives = triplets[..., 1]

    model.eval()
    embeddings_q, embeddings_db = [], []

    with torch.no_grad():
        for img in anchors:
            img_tensor = torch.tensor(img[None, None], dtype=torch.float32).to(device)
            embeddings_q.append(model.encoder(img_tensor))
        for img in positives:
            img_tensor = torch.tensor(img[None, None], dtype=torch.float32).to(device)
            embeddings_db.append(model.encoder(img_tensor))

    embeddings_q = normalize(torch.cat(embeddings_q), dim=1)
    embeddings_db = normalize(torch.cat(embeddings_db), dim=1)

    n_queries = len(embeddings_q)
    recalls = {k: 0 for k in ks}
    aps = {k: [] for k in ks}

    for i in range(n_queries):
        dists = torch.norm(embeddings_db - embeddings_q[i], dim=1)
        sorted_idx = torch.argsort(dists)

        for k in ks:
            if i in sorted_idx[:k].cpu().numpy():
                recalls[k] += 1

            # Compute AP@k
            relevant_positions = (sorted_idx == i).nonzero(as_tuple=True)[0]
            if len(relevant_positions) > 0:
                rank_position = relevant_positions[0].item()
                if rank_position < k:
                    aps[k].append(1.0 / (rank_position + 1))
                else:
                    aps[k].append(0.0)
            else:
                aps[k].append(0.0)

    recall_results = {f"recall@{k}": recalls[k] / n_queries for k in ks}
    map_results = {f"mAP@{k}": float(np.mean(aps[k])) for k in ks}

    return {**recall_results, **map_results}

def save_plot(metric_dict, ylabel, title, save_path):
    """
    Saves a simple training/validation plot.

    Args:
        metric_dict (dict): Dictionary mapping label names to lists of metric values.
        ylabel (str): Label for the y-axis.
        title (str): Plot title.
        save_path (str): Path to save the output image.
    """
    plt.figure(figsize=(8, 5))
    for label, values in metric_dict.items():
        plt.plot(values, label=label, linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_smooth_plot(metric_dict, ylabel, title, save_path, window=5):
    """
    Saves a smoothed, publication-quality training/validation plot using moving averages.

    Args:
        metric_dict (dict): Dictionary mapping label names to lists of metric values.
        ylabel (str): Label for the y-axis.
        title (str): Plot title.
        save_path (str): Path to save the output image.
        window (int): Moving average window size for smoothing.
    """
    plt.figure(figsize=(10, 6))
    for label, values in metric_dict.items():
        smoothed = pd.Series(values).rolling(window=window, min_periods=1).mean()
        plt.plot(smoothed, label=label, linewidth=2)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # High-resolution for publication
    plt.close()

def show_model_summary(model, input_shape=(40, 40), device="cuda"):
    """
    Display a summary of the model architecture using torchsummary.
    Since the TripletNetwork takes 3 inputs, we summarize its encoder.

    Args:
        model (nn.Module): The PatchMatchTripletNetwork model.
        input_shape (tuple): Input image shape, e.g., (40, 40) for grayscale patches.
        device (str): 'cpu' or 'cuda'.
    """
    print("[INFO] Model Architecture:")
    summary(model.encoder, input_size=(1, *input_shape), device=str(device))
