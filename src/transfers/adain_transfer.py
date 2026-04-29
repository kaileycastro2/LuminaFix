"""
AdaIN Style Transfer - Adaptive Instance Normalization.

Based on: "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"
Keras implementation from: https://keras.io/examples/generative/adain/
"""

from typing import Optional
import numpy as np
import logging

from .abstract_transfer import AbstractTransfer
from .base import ProtectionMasks
from .registry import TransferRegistry

logger = logging.getLogger(__name__)

# Lazy imports for TensorFlow/Keras
_tf = None
_keras = None


def _ensure_tensorflow():
    """Lazily import TensorFlow and Keras."""
    global _tf, _keras
    if _tf is None:
        import tensorflow as tf
        _tf = tf
        _keras = tf.keras
        # Suppress TF warnings
        tf.get_logger().setLevel('ERROR')
    return _tf, _keras


def get_mean_std(x, epsilon=1e-5):
    """Compute mean and standard deviation of a tensor."""
    tf, _ = _ensure_tensorflow()
    axes = [1, 2]
    mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
    std = tf.sqrt(variance + epsilon)
    return mean, std


def ada_in(style_features, content_features):
    """
    Adaptive Instance Normalization.

    Aligns the mean and variance of content features with style features.

    Args:
        style_features: Style feature map from encoder
        content_features: Content feature map from encoder

    Returns:
        Transformed feature map
    """
    tf, _ = _ensure_tensorflow()
    content_mean, content_std = get_mean_std(content_features)
    style_mean, style_std = get_mean_std(style_features)

    # Normalize content, then apply style statistics
    normalized = (content_features - content_mean) / content_std
    return style_std * normalized + style_mean


def create_encoder():
    """
    Create VGG19 encoder up to block4_conv1.

    Uses pretrained ImageNet weights (no custom download needed).
    """
    tf, keras = _ensure_tensorflow()

    # Load VGG19 with ImageNet weights
    vgg19 = keras.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(None, None, 3),
    )
    vgg19.trainable = False

    # Extract features up to block4_conv1
    encoder = keras.Model(
        vgg19.input,
        vgg19.get_layer("block4_conv1").output,
        name="adain_encoder"
    )
    return encoder


def create_decoder():
    """
    Create decoder network (mirror of encoder).

    Architecture mirrors VGG19 encoder with upsampling layers.
    """
    tf, keras = _ensure_tensorflow()
    layers = keras.layers

    config = {"kernel_size": 3, "strides": 1, "padding": "same", "activation": "relu"}

    decoder = keras.Sequential([
        layers.InputLayer((None, None, 512)),
        layers.Conv2D(filters=512, **config),
        layers.UpSampling2D(),
        layers.Conv2D(filters=256, **config),
        layers.Conv2D(filters=256, **config),
        layers.Conv2D(filters=256, **config),
        layers.Conv2D(filters=256, **config),
        layers.UpSampling2D(),
        layers.Conv2D(filters=128, **config),
        layers.Conv2D(filters=128, **config),
        layers.UpSampling2D(),
        layers.Conv2D(filters=64, **config),
        layers.Conv2D(
            filters=3,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="sigmoid",
        ),
    ], name="adain_decoder")

    return decoder


@TransferRegistry.register("adain")
class AdaINTransfer(AbstractTransfer):
    """
    AdaIN neural style transfer.

    Uses Adaptive Instance Normalization for real-time
    arbitrary style transfer.

    Features:
    - Real-time style transfer
    - Arbitrary style (no per-style training needed)
    - Uses pretrained VGG19 encoder
    """

    # Max image dimension to prevent OOM errors
    MAX_IMAGE_SIZE = 1024

    def __init__(
        self,
        color_strength: float = 0.7,
        luminance_strength: float = 0.0
    ):
        super().__init__(color_strength, luminance_strength)

        self._encoder = None
        self._decoder = None
        self._style_features = None
        self._original_size = None  # Store original size for upscaling result
        self._tf_available = self._check_tensorflow()

    def _check_tensorflow(self) -> bool:
        """Check if TensorFlow is available."""
        try:
            _ensure_tensorflow()
            return True
        except ImportError:
            return False

    @property
    def method_id(self) -> str:
        return "adain"

    @property
    def method_name(self) -> str:
        return "AdaIN"

    @property
    def method_type(self) -> str:
        return "neural"

    def is_available(self) -> bool:
        """Check if TensorFlow is available."""
        return self._tf_available

    def _load_models(self) -> None:
        """Load encoder and decoder models."""
        if self._encoder is not None:
            return

        tf, keras = _ensure_tensorflow()

        logger.info("Loading AdaIN encoder (VGG19)...")
        self._encoder = create_encoder()

        logger.info("Loading AdaIN decoder...")
        self._decoder = create_decoder()

        # Try to load pretrained decoder weights if available
        try:
            from ..model_manager import get_model_manager
            mm = get_model_manager()
            if mm.is_model_available("adain_decoder"):
                decoder_path = mm.get_model_path("adain_decoder")
                self._decoder.load_weights(str(decoder_path))
                logger.info("Loaded pretrained AdaIN decoder weights")
            else:
                logger.info("Using randomly initialized decoder (no pretrained weights)")
        except Exception as e:
            logger.warning(f"Could not load decoder weights: {e}")

    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image if larger than MAX_IMAGE_SIZE to prevent OOM.

        Args:
            image: BGR image

        Returns:
            Resized BGR image (or original if small enough)
        """
        import cv2
        h, w = image.shape[:2]
        max_dim = max(h, w)

        if max_dim > self.MAX_IMAGE_SIZE:
            scale = self.MAX_IMAGE_SIZE / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h}")

        return image

    def _preprocess_image(self, image: np.ndarray, resize: bool = True) -> np.ndarray:
        """
        Preprocess BGR image for VGG19.

        Args:
            image: BGR uint8 image
            resize: Whether to resize large images

        Returns:
            Preprocessed tensor ready for VGG19
        """
        tf, keras = _ensure_tensorflow()

        # Resize if needed to prevent OOM
        if resize:
            image = self._resize_if_needed(image)

        # BGR to RGB
        rgb = image[:, :, ::-1].copy()

        # Normalize to [0, 1]
        rgb = rgb.astype(np.float32) / 255.0

        # Add batch dimension
        return np.expand_dims(rgb, axis=0)

    def _postprocess_image(self, tensor) -> np.ndarray:
        """
        Convert output tensor back to BGR image.

        Args:
            tensor: Model output tensor [1, H, W, 3]

        Returns:
            BGR uint8 image
        """
        # Remove batch dimension
        if hasattr(tensor, 'numpy'):
            image = tensor.numpy()
        else:
            image = np.array(tensor)

        image = np.squeeze(image, axis=0)

        # Clip to [0, 1] and convert to uint8
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)

        # RGB to BGR
        return image[:, :, ::-1].copy()

    def load_reference(self, image: np.ndarray) -> None:
        """
        Extract style features from reference image.

        Args:
            image: BGR reference image
        """
        if not self._tf_available:
            raise RuntimeError("TensorFlow is not available")

        self._load_models()

        # Preprocess and extract features
        style_input = self._preprocess_image(image)
        self._style_features = self._encoder(style_input, training=False)
        self._reference_loaded = True

        logger.debug(f"Extracted style features shape: {self._style_features.shape}")

    def _apply_transfer(
        self,
        image: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """
        Apply AdaIN style transfer.

        Args:
            image: BGR target image
            strength: Transfer strength (used as alpha for blending)

        Returns:
            Stylized BGR image
        """
        import cv2
        tf, _ = _ensure_tensorflow()

        # Store original size for later upscaling
        original_h, original_w = image.shape[:2]

        # Preprocess content image (includes resizing if needed)
        content_input = self._preprocess_image(image)

        # Extract content features
        content_features = self._encoder(content_input, training=False)

        # Apply AdaIN transform
        # Use strength as interpolation factor
        if strength < 1.0:
            # Interpolate between content and style features
            t = ada_in(self._style_features, content_features)
            t = strength * t + (1 - strength) * content_features
        else:
            t = ada_in(self._style_features, content_features)

        # Decode to image
        output = self._decoder(t, training=False)

        # Postprocess
        result = self._postprocess_image(output)

        # Resize back to original size if it was downscaled
        if result.shape[0] != original_h or result.shape[1] != original_w:
            result = cv2.resize(result, (original_w, original_h), interpolation=cv2.INTER_CUBIC)

        return result
