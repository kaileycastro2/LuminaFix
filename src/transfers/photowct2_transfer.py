"""
PhotoWCT2 Style Transfer - Photorealistic style transfer via wavelet transforms.

Implements whitening and coloring transform in VGG feature space
for photorealistic color/style transfer.
"""

from typing import Optional, Tuple
import numpy as np
import logging

from .abstract_transfer import AbstractTransfer
from .base import ProtectionMasks
from .registry import TransferRegistry

logger = logging.getLogger(__name__)

# Lazy imports for PyTorch
_torch = None
_nn = None
_F = None


def _ensure_torch():
    """Lazily import PyTorch."""
    global _torch, _nn, _F
    if _torch is None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        _torch = torch
        _nn = nn
        _F = F
    return _torch, _nn, _F


def create_vgg_encoder():
    """Create VGG-19 encoder for feature extraction."""
    torch, nn, F = _ensure_torch()

    class VGGEncoder(nn.Module):
        """VGG-19 encoder up to relu4_1."""

        def __init__(self):
            super().__init__()

            # VGG configuration for encoder
            # Conv layers: (in_channels, out_channels)
            vgg_config = [
                (3, 64), (64, 64), 'M',      # Block 1
                (64, 128), (128, 128), 'M',  # Block 2
                (128, 256), (256, 256), (256, 256), (256, 256), 'M',  # Block 3
                (256, 512), (512, 512), (512, 512), (512, 512),       # Block 4 (no pool)
            ]

            layers = []
            for x in vgg_config:
                if x == 'M':
                    layers.append(nn.MaxPool2d(2, 2))
                else:
                    in_c, out_c = x
                    layers.append(nn.Conv2d(in_c, out_c, 3, padding=1))
                    layers.append(nn.ReLU(inplace=True))

            self.features = nn.Sequential(*layers)

        def forward(self, x):
            return self.features(x)

    return VGGEncoder()


def create_decoder():
    """Create decoder network (inverse of VGG encoder)."""
    torch, nn, F = _ensure_torch()

    class Decoder(nn.Module):
        """Decoder to reconstruct image from VGG features."""

        def __init__(self):
            super().__init__()

            # Mirror of VGG encoder
            decoder_config = [
                (512, 512), (512, 512), (512, 512), (512, 256), 'U',  # Block 4
                (256, 256), (256, 256), (256, 256), (256, 128), 'U',  # Block 3
                (128, 128), (128, 64), 'U',                           # Block 2
                (64, 64), (64, 3),                                     # Block 1
            ]

            layers = []
            for x in decoder_config:
                if x == 'U':
                    layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
                else:
                    in_c, out_c = x
                    layers.append(nn.ReflectionPad2d(1))
                    layers.append(nn.Conv2d(in_c, out_c, 3))
                    if out_c != 3:  # No ReLU on final layer
                        layers.append(nn.ReLU(inplace=True))

            self.decode = nn.Sequential(*layers)

        def forward(self, x):
            return self.decode(x)

    return Decoder()


def whitening_coloring_transform(content_feat, style_feat, alpha=1.0):
    """
    Whitening and Coloring Transform (WCT).

    1. Whiten content features (remove correlations)
    2. Color with style statistics

    Args:
        content_feat: Content features [B, C, H, W]
        style_feat: Style features [B, C, H, W]
        alpha: Blending factor

    Returns:
        Transformed features
    """
    torch, _, _ = _ensure_torch()

    B, C, H, W = content_feat.size()

    # Reshape to [B, C, H*W]
    content_flat = content_feat.view(B, C, -1)
    style_flat = style_feat.view(B, C, -1)

    # Compute means
    content_mean = content_flat.mean(dim=2, keepdim=True)
    style_mean = style_flat.mean(dim=2, keepdim=True)

    # Center the features
    content_centered = content_flat - content_mean
    style_centered = style_flat - style_mean

    # Compute covariance matrices
    content_cov = torch.bmm(content_centered, content_centered.transpose(1, 2)) / (H * W - 1)
    style_cov = torch.bmm(style_centered, style_centered.transpose(1, 2)) / (H * W - 1)

    # Add small epsilon for numerical stability
    eye = torch.eye(C, device=content_feat.device).unsqueeze(0) * 1e-5
    content_cov = content_cov + eye
    style_cov = style_cov + eye

    # Whitening transform using SVD
    try:
        U_c, S_c, V_c = torch.svd(content_cov)
        U_s, S_s, V_s = torch.svd(style_cov)

        # Whitening matrix: D^(-1/2) * U^T
        k_c = C
        k_s = C

        # Build whitening matrix
        S_c_inv_sqrt = torch.diag_embed(S_c[:, :k_c].pow(-0.5))
        whiten_matrix = torch.bmm(torch.bmm(U_c[:, :, :k_c], S_c_inv_sqrt), U_c[:, :, :k_c].transpose(1, 2))

        # Coloring matrix: U * D^(1/2)
        S_s_sqrt = torch.diag_embed(S_s[:, :k_s].pow(0.5))
        color_matrix = torch.bmm(torch.bmm(U_s[:, :, :k_s], S_s_sqrt), U_s[:, :, :k_s].transpose(1, 2))

        # Apply transforms
        whitened = torch.bmm(whiten_matrix, content_centered)
        colored = torch.bmm(color_matrix, whitened)

        # Add style mean
        transformed = colored + style_mean

        # Blend with original content
        if alpha < 1.0:
            transformed = alpha * transformed + (1 - alpha) * content_flat

        return transformed.view(B, C, H, W)

    except RuntimeError as e:
        logger.warning(f"WCT SVD failed: {e}. Falling back to mean-only transfer.")
        # Fallback: just shift mean
        transformed = content_centered + style_mean
        return transformed.view(B, C, H, W)


@TransferRegistry.register("photowct2")
class PhotoWCT2Transfer(AbstractTransfer):
    """
    PhotoWCT2 photorealistic style transfer.

    Uses VGG encoder-decoder with whitening and coloring
    transform for photorealistic color transfer.

    Features:
    - Neural network-based
    - Photorealistic output (preserves structure)
    - GPU accelerated (if available)
    """

    def __init__(
        self,
        color_strength: float = 0.7,
        luminance_strength: float = 0.0
    ):
        super().__init__(color_strength, luminance_strength)

        self._encoder = None
        self._decoder = None
        self._device = None
        self._style_features = None
        self._torch_available = self._check_torch()

    def _check_torch(self) -> bool:
        """Check if PyTorch is available."""
        try:
            _ensure_torch()
            return True
        except ImportError:
            return False

    @property
    def method_id(self) -> str:
        return "photowct2"

    @property
    def method_name(self) -> str:
        return "PhotoWCT2"

    @property
    def method_type(self) -> str:
        return "neural"

    def is_available(self) -> bool:
        """Check if PhotoWCT2 is available (disabled - no PyTorch weights exist)."""
        # Disabled: PhotoWCT2 repo only has TensorFlow weights, not PyTorch
        # Also reduces memory usage which causes OOM on CPU
        return False

    def _load_models(self) -> None:
        """Load encoder and decoder models."""
        if self._encoder is not None:
            return

        torch, nn, F = _ensure_torch()

        # Get device
        from ..device_manager import get_device_manager
        dm = get_device_manager()
        self._device = dm.device

        # Create models
        self._encoder = create_vgg_encoder()
        self._decoder = create_decoder()

        self._encoder.to(self._device)
        self._decoder.to(self._device)

        self._encoder.eval()
        self._decoder.eval()

        # Try to load pretrained weights
        try:
            from ..model_manager import get_model_manager
            mm = get_model_manager()

            # Load VGG weights for encoder (from torchvision if available)
            try:
                from torchvision.models import vgg19, VGG19_Weights
                vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
                # Copy weights to our encoder
                self._copy_vgg_weights(vgg)
                logger.info("Loaded VGG-19 pretrained weights for encoder")
            except Exception as e:
                logger.warning(f"Could not load VGG weights: {e}")

            # Try to load decoder weights
            try:
                decoder_path = mm.get_model_path("photowct2_decoder")
                state_dict = torch.load(decoder_path, map_location=self._device)
                self._decoder.load_state_dict(state_dict, strict=False)
                logger.info("Loaded PhotoWCT2 decoder weights")
            except Exception as e:
                logger.warning(f"Could not load decoder weights: {e}. Using random init.")

        except Exception as e:
            logger.warning(f"Model loading error: {e}")

    def _copy_vgg_weights(self, vgg_model) -> None:
        """Copy weights from torchvision VGG to our encoder."""
        torch, _, _ = _ensure_torch()

        vgg_layers = list(vgg_model.features.children())
        encoder_layers = list(self._encoder.features.children())

        vgg_idx = 0
        for i, layer in enumerate(encoder_layers):
            if isinstance(layer, torch.nn.Conv2d):
                while vgg_idx < len(vgg_layers) and not isinstance(vgg_layers[vgg_idx], torch.nn.Conv2d):
                    vgg_idx += 1
                if vgg_idx < len(vgg_layers):
                    layer.weight.data = vgg_layers[vgg_idx].weight.data.clone()
                    layer.bias.data = vgg_layers[vgg_idx].bias.data.clone()
                    vgg_idx += 1

    def _preprocess_image(self, image: np.ndarray):
        """Convert BGR numpy array to normalized torch tensor."""
        torch, _, _ = _ensure_torch()

        # BGR to RGB
        rgb = image[:, :, ::-1].copy().astype(np.float32)

        # Normalize with ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406]) * 255
        std = np.array([0.229, 0.224, 0.225]) * 255
        rgb = (rgb - mean) / std

        # HWC to CHW
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor.to(self._device)

    def _postprocess_image(self, tensor) -> np.ndarray:
        """Convert torch tensor back to BGR numpy array."""
        torch, _, _ = _ensure_torch()

        # Remove batch dimension, move to CPU
        tensor = tensor.squeeze(0).cpu()

        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1) * 255
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) * 255
        tensor = tensor * std + mean

        # Clamp to [0, 255]
        tensor = tensor.clamp(0, 255)

        # CHW to HWC
        image = tensor.numpy().transpose(1, 2, 0).astype(np.uint8)

        # RGB to BGR
        return image[:, :, ::-1].copy()

    def load_reference(self, image: np.ndarray) -> None:
        """
        Extract style features from reference image.

        Args:
            image: BGR reference image
        """
        if not self._torch_available:
            raise RuntimeError("PyTorch is not available")

        torch, _, _ = _ensure_torch()

        self._load_models()

        # Preprocess and extract features
        style_tensor = self._preprocess_image(image)
        with torch.no_grad():
            self._style_features = self._encoder(style_tensor)

        self._reference_loaded = True

    def _apply_transfer(
        self,
        image: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """
        Apply PhotoWCT2 style transfer.

        Args:
            image: BGR target image
            strength: Transfer strength (alpha for WCT)

        Returns:
            Transferred BGR image
        """
        torch, _, _ = _ensure_torch()

        # Preprocess content
        content_tensor = self._preprocess_image(image)

        with torch.no_grad():
            # Encode content
            content_features = self._encoder(content_tensor)

            # Apply WCT
            transformed = whitening_coloring_transform(
                content_features,
                self._style_features,
                alpha=strength
            )

            # Decode
            output = self._decoder(transformed)

        # Postprocess
        result = self._postprocess_image(output)

        return result
