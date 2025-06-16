import numpy as np
import tensorflow as tf

class PatchPairDatasetLoader:
    """
    Loads patch pair datasets stored in .npz format and prepares them for training
    with Siamese networks using TensorFlow.
    """

    def __init__(self, npz_path, batch_size=64, shuffle=True):
        """
        Initializes the dataset loader.

        Args:
            npz_path (str): Path to the .npz file containing 'patches' and 'labels'.
            batch_size (int): Batch size for training.
            shuffle (bool): Whether to shuffle the dataset.
        """
        self.npz_path = npz_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = None

    def _load_data(self):
        """
        Loads the data from the .npz file and prepares patch1, patch2, and labels.

        Returns:
            tuple: (patch1, patch2, labels)
        """
        data = np.load(self.npz_path)
        patches = data['patches']  # shape: (N, H, W, 2)
        labels = data['labels']    # shape: (N,)

        patches = patches.astype(np.float32) / 255.0
        patch1 = patches[..., 0][..., np.newaxis]  # shape: (N, H, W, 1)
        patch2 = patches[..., 1][..., np.newaxis]  # shape: (N, H, W, 1)

        labels = labels.astype(np.float32)
        labels[labels == -1] = 0  # Treat hard negatives as negatives for contrastive loss

        return patch1, patch2, labels

    def get_dataset(self):
        """
        Returns the TensorFlow Dataset object.

        Returns:
            tf.data.Dataset: A dataset yielding ((patch1, patch2), label) tuples.
        """
        if self.dataset is not None:
            return self.dataset

        patch1, patch2, labels = self._load_data()
        dataset = tf.data.Dataset.from_tensor_slices(((patch1, patch2), labels))

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(labels))

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        self.dataset = dataset
        return dataset
