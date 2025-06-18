import os
import cv2
import numpy as np
import random
from tqdm import tqdm

class GeneratePatches:
    """
    Generates triplet image patches (anchor, positive, negative) from HPatches dataset sequences.

    The patches are sampled from a grid on a reference image, warped using homographies to obtain
    positives, and shifted to generate negatives (both random and hard negatives).

    Attributes:
        folder_path (str): Path to HPatches dataset root.
        save_path (str): Destination path to save generated triplets as a .npz file.
        allowed_sequences (list): Subset of sequence names to process.
        dataset_size (int): Number of triplet samples to generate.
        patch_size (int): Size (in pixels) of each square patch.
        grid_spacing (int): Step size for grid sampling on the reference image.
    """

    def __init__(self, folder_path, save_path, allowed_sequences=None,
                 dataset_size=50000, patch_size=40, grid_spacing=20):
        self.folder_path = folder_path
        self.save_path = save_path
        self.allowed_sequences = allowed_sequences
        self.dataset_size = dataset_size
        self.patch_size = patch_size
        self.grid_spacing = grid_spacing

        self.triplets = []
        self.hard_negative_count = 0
        self.random_negative_count = 0

    def extract(self):
        """
        Main loop to process HPatches sequences and extract triplet patches.

        Uses a grid over image 1 in each sequence, samples a patch, computes its warped
        correspondence using homography, and generates positive and negative patches.

        Triplets are saved in memory and later written to disk.
        """
        all_dirs = sorted([d for d in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, d))])
        sequences = self.allowed_sequences if self.allowed_sequences else all_dirs
        total_sequences = len(sequences)
        samples_per_sequence = self.dataset_size // total_sequences

        print(f"[INFO] Using {total_sequences} sequences")

        for subdir in tqdm(sequences, desc="[INFO] Processing sequences"):
            seq_path = os.path.join(self.folder_path, subdir)
            images, homographies = self._load_images_and_homographies(seq_path)

            if len(images) < 6:
                continue

            count = 0
            for i in range(5):  # H_1_2 to H_1_6
                img1 = images[0]  # anchor image
                img2 = images[i + 1]  # target image
                H = homographies.get(i)
                if H is None:
                    continue

                height, width = img1.shape
                for y in range(self.patch_size // 2, height - self.patch_size // 2, self.grid_spacing):
                    for x in range(self.patch_size // 2, width - self.patch_size // 2, self.grid_spacing):
                        pt1 = np.array([x, y, 1])
                        pt2 = H @ pt1
                        pt2 /= pt2[2]

                        if not self._is_within_bounds(pt2[:2], img2.shape):
                            continue

                        anchor = self._extract_patch(img1, pt1[:2])
                        positive = self._extract_patch(img2, pt2[:2])
                        if anchor is None or positive is None:
                            continue

                        neg = self._get_offset_patch(img2, pt2, offset_range=(20, 40))
                        if neg is not None:
                            self.triplets.append(np.stack([anchor, positive, neg], axis=-1))
                            self.random_negative_count += 1

                        hard_neg = self._get_offset_patch(img2, pt2, offset_range=(10, 15))
                        if hard_neg is not None:
                            self.triplets.append(np.stack([anchor, positive, hard_neg], axis=-1))
                            self.hard_negative_count += 1

                        count += 2
                        if count >= samples_per_sequence:
                            break
                    if count >= samples_per_sequence:
                        break

        self._save_dataset()

    def _load_images_and_homographies(self, seq_path):
        """
        Load grayscale images and homography matrices for a given HPatches sequence.

        Args:
            seq_path (str): Path to one HPatches sequence directory.

        Returns:
            images (list of ndarray): List of grayscale images.
            homographies (dict): Dictionary of homographies from image 1 to images 2 to 6.
        """
        images = []
        homographies = {}

        for i in range(1, 7):
            img_path = os.path.join(seq_path, f"{i}.ppm")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)

        for i in range(1, 6):
            h_path = os.path.join(seq_path, f"H_1_{i + 1}")
            if os.path.exists(h_path):
                with open(h_path, 'r') as f:
                    matrix = [list(map(float, line.strip().split())) for line in f if line.strip()]
                    homographies[i - 1] = np.array(matrix)

        return images, homographies

    def _is_within_bounds(self, pt, shape):
        """
        Check whether a patch centered at point `pt` lies fully within image bounds.

        Args:
            pt (tuple): (x, y) coordinates.
            shape (tuple): Image shape (height, width).

        Returns:
            bool: True if patch fits within image, else False.
        """
        x, y = int(pt[0]), int(pt[1])
        return (self.patch_size // 2 <= x < shape[1] - self.patch_size // 2 and
                self.patch_size // 2 <= y < shape[0] - self.patch_size // 2)

    def _extract_patch(self, img, pt):
        """
        Extract square patch centered at given point.

        Args:
            img (ndarray): Grayscale image.
            pt (tuple): Center (x, y) coordinates.

        Returns:
            patch (ndarray or None): Extracted patch or None if invalid.
        """
        x, y = int(round(pt[0])), int(round(pt[1]))
        half = self.patch_size // 2
        return img[y - half:y + half, x - half:x + half]

    def _get_offset_patch(self, img, ref_pt, offset_range):
        """
        Generate negative patch by shifting reference point with a random offset.

        Args:
            img (ndarray): Target image.
            ref_pt (ndarray): Homogeneous coordinates of a point.
            offset_range (tuple): (min_offset, max_offset) in pixels.

        Returns:
            patch (ndarray or None): Distractor patch or None if out of bounds.
        """
        pt = ref_pt.copy()
        pt[0] += random.choice([-1, 1]) * random.randint(*offset_range)
        pt[1] += random.choice([-1, 1]) * random.randint(*offset_range)
        if self._is_within_bounds(pt[:2], img.shape):
            return self._extract_patch(img, pt[:2])
        return None

    def _save_dataset(self):
        """
        Save triplets to .npz file on disk.
        """
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        np.savez_compressed(self.save_path, triplets=np.array(self.triplets))

        print(f"\n[INFO] Triplet dataset saved to {self.save_path}")
        print(f"   Total triplets         : {len(self.triplets)}")
        print(f"   - Negatives            : {self.random_negative_count}")
        print(f"   - Hard Negatives       : {self.hard_negative_count}")