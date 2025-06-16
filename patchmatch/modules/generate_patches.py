import os
import cv2
import numpy as np
import random
from tqdm import tqdm


class GeneratePatches:
    """
    Patch generator using grid-based sampling and ground-truth homographies from the HPatches dataset.
    It creates:
        - Positive pairs (true matches via homography)
        - Negative pairs (random spatial offsets)
        - Hard negative pairs (close but incorrect patches)
    The output is saved as a compressed .npz file containing patch pairs and their labels.
    """

    def __init__(self, folder_path, save_path, dataset_size=50000, patch_size=40, grid_spacing=20):
        """
        Initialize the patch generator.

        Args:
            folder_path (str): Path to the HPatches dataset root.
            save_path (str): Path to save the generated .npz file.
            dataset_size (int): Total number of patch pairs to generate.
            patch_size (int): Size (pixels) of each square patch.
            grid_spacing (int): Distance (in pixels) between grid sampling points.
        """
        self.folder_path = folder_path
        self.save_path = save_path
        self.dataset_size = dataset_size
        self.patch_size = patch_size
        self.grid_spacing = grid_spacing
        self.X = []
        self.y = []

        self.positive_count = 0
        self.negative_count = 0
        self.hard_negative_count = 0

    def extract(self):
        """
        Extract patches from all sequences in the dataset using a uniform grid,
        apply homographies to find correspondences, and generate positive, negative,
        and hard negative patch pairs. Save results when finished.
        """
        subdirs = [d for d in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, d))]
        total_sequences = len(subdirs)
        samples_per_sequence = self.dataset_size // total_sequences

        print(f"[INFO] Using {total_sequences} valid sequences out of {total_sequences} total.")

        for subdir in tqdm(subdirs, desc="[INFO] Processing sequences"):
            seq_path = os.path.join(self.folder_path, subdir)
            images = []
            homographies = {}

            # Load 6 grayscale images per sequence
            for i in range(1, 7):
                img_path = os.path.join(seq_path, f"{i}.ppm")
                if not os.path.exists(img_path):
                    break
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    break
                images.append(img)

            # Load homographies H_1_{i+1}
            for i in range(1, 6):
                H_path = os.path.join(seq_path, f"H_1_{i + 1}")
                if os.path.exists(H_path):
                    with open(H_path, 'r') as f:
                        matrix = [list(map(float, line.strip().split())) for line in f if line.strip()]
                        homographies[i - 1] = np.array(matrix)

            if len(images) < 6:
                continue

            count = 0
            for i in range(5):
                img1 = images[0]
                img2 = images[i + 1]
                H = homographies.get(i, None)
                if H is None:
                    continue

                height, width = img1.shape
                for y in range(self.patch_size//2, height - self.patch_size//2, self.grid_spacing):
                    for x in range(self.patch_size//2, width - self.patch_size//2, self.grid_spacing):
                        pt1 = np.array([x, y, 1])
                        pt2 = H @ pt1
                        pt2 /= pt2[2]

                        if not self._is_within_bounds(pt2[:2], img2.shape):
                            continue

                        patch1 = self._extract_patch(img1, pt1[:2])
                        patch2 = self._extract_patch(img2, pt2[:2])

                        if patch1 is not None and patch2 is not None:
                            self.X.append(np.stack([patch1, patch2], axis=-1))
                            self.y.append(1)  # Positive
                            self.positive_count += 1
                            count += 1

                        # Negative pair (random offset)
                        pt2_bad = pt2.copy()
                        pt2_bad[0] += random.choice([-1, 1]) * random.randint(20, 40)
                        pt2_bad[1] += random.choice([-1, 1]) * random.randint(20, 40)

                        if self._is_within_bounds(pt2_bad[:2], img2.shape):
                            patch2_bad = self._extract_patch(img2, pt2_bad[:2])
                            if patch2_bad is not None:
                                self.X.append(np.stack([patch1, patch2_bad], axis=-1))
                                self.y.append(0)  # Negative
                                self.negative_count += 1
                                count += 1

                        # Hard negative pair (close offset)
                        pt2_hard = pt2.copy()
                        pt2_hard[0] += random.choice([-1, 1]) * random.randint(10, 15)
                        pt2_hard[1] += random.choice([-1, 1]) * random.randint(10, 15)

                        if self._is_within_bounds(pt2_hard[:2], img2.shape):
                            patch2_hard = self._extract_patch(img2, pt2_hard[:2])
                            if patch2_hard is not None:
                                self.X.append(np.stack([patch1, patch2_hard], axis=-1))
                                self.y.append(-1)  # Hard negative
                                self.hard_negative_count += 1
                                count += 1

                        if count >= samples_per_sequence:
                            break
                    if count >= samples_per_sequence:
                        break

        self._save_dataset()

    def _is_within_bounds(self, pt, shape):
        """
        Check if a patch centered at pt would be fully within image bounds.

        Args:
            pt (tuple or ndarray): Center point (x, y).
            shape (tuple): Shape of the image (height, width).

        Returns:
            bool: True if patch is valid, False otherwise.
        """
        x, y = int(pt[0]), int(pt[1])
        return (self.patch_size // 2 <= x < shape[1] - self.patch_size // 2 and
                self.patch_size // 2 <= y < shape[0] - self.patch_size // 2)

    def _extract_patch(self, img, pt):
        """
        Extract a square patch from an image centered at pt.

        Args:
            img (ndarray): Grayscale image.
            pt (ndarray): Center point (x, y).

        Returns:
            ndarray: Extracted patch of shape (patch_size, patch_size) or None if out-of-bounds.
        """
        x, y = int(round(pt[0])), int(round(pt[1]))
        half = self.patch_size // 2
        return img[y - half:y + half, x - half:x + half]

    def _save_dataset(self):
        """
        Save the dataset as a compressed .npz file.
        """
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        np.savez_compressed(self.save_path, patches=np.array(self.X), labels=np.array(self.y))
        print(f"\n[INFO] Dataset saved to {self.save_path}")
        print(f"   Total samples         : {len(self.y)}")
        print(f"   - Positive pairs      : {self.positive_count}")
        print(f"   - Negative pairs      : {self.negative_count}")
        print(f"   - Hard negative pairs : {self.hard_negative_count}")
