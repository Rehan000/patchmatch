import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

class GeneratePatches:
    """
    Generates triplet image patches (anchor, positive, negative) from HPatches dataset sequences.

    The patches are sampled from a grid on a reference image, warped using homographies to obtain
    positives, and shifted to generate negatives (both random and hard negatives). SSIM filtering
    is optionally applied to retain only high-similarity distractors as hard negatives.

    Attributes:
        folder_path (str): Path to HPatches dataset root.
        save_path (str): Destination path to save generated triplets as a .npz file.
        allowed_sequences (list): Subset of sequence names to process.
        dataset_size (int): Number of triplet samples to generate.
        patch_size (int): Size (in pixels) of each square patch.
        grid_spacing (int): Step size for grid sampling on the reference image.
        use_ssim_filter (bool): Whether to use SSIM filtering for hard negatives.
        ssim_threshold (float): Threshold to accept hard negatives based on SSIM score.
    """

    def __init__(self, folder_path, save_path, allowed_sequences=None,
                 dataset_size=50000, patch_size=40, grid_spacing=20,
                 use_ssim_filter=True, ssim_threshold=0.5):
        self.folder_path = folder_path
        self.save_path = save_path
        self.allowed_sequences = allowed_sequences
        self.dataset_size = dataset_size
        self.patch_size = patch_size
        self.grid_spacing = grid_spacing
        self.use_ssim_filter = use_ssim_filter
        self.ssim_threshold = ssim_threshold

        self.triplets = []
        self.hard_negative_count = 0
        self.random_negative_count = 0

    def extract(self):
        """
        Main function to generate triplet patches from HPatches sequences.
        Iterates over grid locations in reference images, generates anchors and positives
        using homographies, and samples random and hard negatives. Optionally filters
        hard negatives using SSIM similarity.
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
            for i in range(5):
                img1 = images[0]
                img2 = images[i + 1]
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
                            if self.use_ssim_filter:
                                score = self._ssim(anchor, hard_neg)
                                if score < self.ssim_threshold:
                                    continue
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
        Load grayscale images and corresponding homography matrices for a given sequence.

        Args:
            seq_path (str): Path to sequence directory.

        Returns:
            images (list): List of grayscale images.
            homographies (dict): Mapping from index to homography matrices.
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
        Check if a patch centered at a given point lies fully within the image boundaries.

        Args:
            pt (tuple): (x, y) coordinates.
            shape (tuple): Shape of the image (height, width).

        Returns:
            bool: True if patch lies within bounds, else False.
        """
        x, y = int(pt[0]), int(pt[1])
        return (self.patch_size // 2 <= x < shape[1] - self.patch_size // 2 and
                self.patch_size // 2 <= y < shape[0] - self.patch_size // 2)

    def _extract_patch(self, img, pt):
        """
        Extract a square patch centered at a specific point in the image.

        Args:
            img (ndarray): Grayscale image.
            pt (tuple): (x, y) coordinates.

        Returns:
            ndarray or None: Extracted patch or None if coordinates are out of bounds.
        """
        x, y = int(round(pt[0])), int(round(pt[1]))
        half = self.patch_size // 2
        return img[y - half:y + half, x - half:x + half]

    def _get_offset_patch(self, img, ref_pt, offset_range):
        """
        Generate a patch at a random offset from a reference point.

        Args:
            img (ndarray): Target image.
            ref_pt (ndarray): Reference homogeneous coordinates.
            offset_range (tuple): Range of random offsets.

        Returns:
            ndarray or None: Offset patch or None if out of bounds.
        """
        pt = ref_pt.copy()
        pt[0] += random.choice([-1, 1]) * random.randint(*offset_range)
        pt[1] += random.choice([-1, 1]) * random.randint(*offset_range)
        if self._is_within_bounds(pt[:2], img.shape):
            return self._extract_patch(img, pt[:2])
        return None

    def _ssim(self, patch1, patch2):
        """
        Compute Structural Similarity Index (SSIM) between two patches.

        Args:
            patch1 (ndarray): First patch.
            patch2 (ndarray): Second patch.

        Returns:
            float: SSIM score between 0 and 1.
        """
        try:
            return ssim(patch1, patch2)
        except:
            return 0.0

    def _save_dataset(self):
        """
        Save all generated triplets to a compressed .npz file on disk.
        """
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        np.savez_compressed(self.save_path, triplets=np.array(self.triplets))

        print(f"\n[INFO] Triplet dataset saved to {self.save_path}")
        print(f"   Total triplets         : {len(self.triplets)}")
        print(f"   - Negatives            : {self.random_negative_count}")
        print(f"   - Hard Negatives       : {self.hard_negative_count}")
