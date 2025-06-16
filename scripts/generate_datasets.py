import os
import yaml
from patchmatch import GeneratePatches

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config('config/config.yaml')

    base_path = os.path.abspath(config['dataset_root'])
    output_path = os.path.abspath(config['dataset_output_dir'])

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"[ERROR] Dataset folder not found at: {base_path}")

    os.makedirs(output_path, exist_ok=True)

    splits = config.get('splits', {})
    patch_size = config.get('patch_size', 40)
    grid_spacing = config.get('grid_spacing', 20)

    for split, size in splits.items():
        print(f"\n[INFO] Generating {split}_dataset.npz with {size} samples...")
        generator = GeneratePatches(
            folder_path=base_path,
            save_path=os.path.join(output_path, f"{split}_dataset.npz"),
            dataset_size=size,
            patch_size=patch_size,
            grid_spacing=grid_spacing
        )
        generator.extract()

if __name__ == "__main__":
    main()
