import os
import yaml
import random
from patchmatch import GeneratePatches

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def generate_sequence_splits(base_path, split_ratios=(0.80, 0.10, 0.10), seed=42):
    """
    Generate train/valid/test splits from available sequence directories.

    Args:
        base_path (str): Root directory containing HPatches sequences.
        split_ratios (tuple): Ratios for (train, valid, test) splits.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Keys are 'train', 'valid', 'test' and values are lists of sequence names.
    """
    all_sequences = sorted([d for d in os.listdir(base_path)
                            if os.path.isdir(os.path.join(base_path, d))])

    random.seed(seed)
    random.shuffle(all_sequences)

    n_total = len(all_sequences)
    n_train = int(split_ratios[0] * n_total)
    n_valid = int(split_ratios[1] * n_total)

    return {
        "train": all_sequences[:n_train],
        "valid": all_sequences[n_train:n_train + n_valid],
        "test": all_sequences[n_train + n_valid:]
    }

def main():
    config = load_config('config/config.yaml')

    base_path = os.path.abspath(config['dataset_root'])
    output_path = os.path.abspath(config['dataset_output_dir'])

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"[ERROR] Dataset folder not found at: {base_path}")

    os.makedirs(output_path, exist_ok=True)

    splits_config = config.get('splits', {})
    patch_size = config.get('patch_size', 40)
    grid_spacing = config.get('grid_spacing', 20)

    split_sequences = generate_sequence_splits(base_path)

    for split, size in splits_config.items():
        if split not in split_sequences:
            continue

        print(f"\n[INFO] Generating {split}_dataset.npz with {size} samples...")
        generator = GeneratePatches(
            folder_path=base_path,
            save_path=os.path.join(output_path, f"{split}_dataset.npz"),
            allowed_sequences=split_sequences[split],
            dataset_size=size,
            patch_size=patch_size,
            grid_spacing=grid_spacing
        )
        generator.extract()

if __name__ == "__main__":
    main()
