import tensorflow as tf
from tensorflow.keras import layers, models

class PatchMatchSiameseDescriptor:
    """
    Class to build and manage a lightweight Siamese descriptor model with contrastive loss.
    """

    def __init__(self, input_shape=(40, 40, 1), embedding_dim=64, margin=1.0):
        """
        Initializes the model components.

        Args:
            input_shape (tuple): Shape of input patch (default is 40x40 grayscale).
            embedding_dim (int): Length of the learned descriptor vector.
            margin (float): Margin for contrastive loss.
        """
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.margin = margin

        self.encoder = self._build_encoder()
        self.siamese_model = self._build_siamese_model()

    def _build_encoder(self):
        """
        Builds the encoder model using depthwise separable convolutions.

        Returns:
            tf.keras.Model: Encoder model.
        """
        inputs = layers.Input(shape=self.input_shape)

        x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.DepthwiseConv2D(3, padding='same', activation='relu')(x)
        x = layers.Conv2D(64, 1, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)  # Output: 20x20

        x = layers.DepthwiseConv2D(3, padding='same', activation='relu')(x)
        x = layers.Conv2D(64, 1, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(self.embedding_dim)(x)
        x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)

        return models.Model(inputs, x, name="LightweightDescriptor")

    def _build_siamese_model(self):
        """
        Builds the full Siamese model using the shared encoder.

        Returns:
            tf.keras.Model: Siamese model.
        """
        input_a = layers.Input(shape=self.input_shape)
        input_b = layers.Input(shape=self.input_shape)

        encoded_a = self.encoder(input_a)
        encoded_b = self.encoder(input_b)

        distance = layers.Lambda(
            lambda tensors: tf.norm(tensors[0] - tensors[1], axis=1, keepdims=True)
        )([encoded_a, encoded_b])

        return models.Model(inputs=[input_a, input_b], outputs=distance, name="SiameseNetwork")

    def get_model(self):
        """
        Returns the Siamese model.

        Returns:
            tf.keras.Model
        """
        return self.siamese_model

    def get_encoder(self):
        """
        Returns the encoder model (can be used standalone for descriptors).

        Returns:
            tf.keras.Model
        """
        return self.encoder

    def get_contrastive_loss(self):
        """
        Returns the contrastive loss function.

        Returns:
            Callable: A Keras-compatible loss function.
        """
        def loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            return tf.reduce_mean(
                y_true * tf.square(y_pred) +
                (1 - y_true) * tf.square(tf.maximum(self.margin - y_pred, 0))
            )
        return loss
