"""
MSG-Net Style Transfer - Multi-Style Generative Network.

Neural network-based style transfer that can handle multiple styles
in a single forward pass.
"""

from typing import Optional
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


class GramMatrix(_ensure_torch()[1].Module if _ensure_torch()[0] else object):
    """Compute Gram matrix for style features."""

    def forward(self, x):
        torch, nn, F = _ensure_torch()
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div_(h * w)


class ConvLayer:
    """Convolution layer with reflection padding."""

    @staticmethod
    def create(in_channels, out_channels, kernel_size, stride):
        torch, nn, F = _ensure_torch()
        padding = kernel_size // 2
        return nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        )


class UpsampleConvLayer:
    """Upsampling followed by convolution."""

    @staticmethod
    def create(in_channels, out_channels, kernel_size, stride, upsample=None):
        torch, nn, F = _ensure_torch()
        layers = []
        if upsample:
            layers.append(nn.Upsample(scale_factor=upsample, mode='nearest'))
        padding = kernel_size // 2
        layers.append(nn.ReflectionPad2d(padding))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
        return nn.Sequential(*layers)


class ResidualBlock:
    """Residual block with two convolutions."""

    @staticmethod
    def create(channels):
        torch, nn, F = _ensure_torch()

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = ConvLayer.create(channels, channels, 3, 1)
                self.in1 = nn.InstanceNorm2d(channels)
                self.conv2 = ConvLayer.create(channels, channels, 3, 1)
                self.in2 = nn.InstanceNorm2d(channels)
                self.relu = nn.ReLU()

            def forward(self, x):
                residual = x
                out = self.relu(self.in1(self.conv1(x)))
                out = self.in2(self.conv2(out))
                return out + residual

        return Block()


def create_msgnet_model():
    """Create MSG-Net model architecture."""
    torch, nn, F = _ensure_torch()

    class MSGNet(nn.Module):
        """Multi-Style Generative Network."""

        def __init__(self, input_nc=3, output_nc=3, ngf=128, n_blocks=6):
            super().__init__()

            # Initial convolution
            self.conv1 = ConvLayer.create(input_nc, ngf, 9, 1)
            self.in1 = nn.InstanceNorm2d(ngf)

            # Downsampling
            self.conv2 = ConvLayer.create(ngf, ngf * 2, 3, 2)
            self.in2 = nn.InstanceNorm2d(ngf * 2)
            self.conv3 = ConvLayer.create(ngf * 2, ngf * 4, 3, 2)
            self.in3 = nn.InstanceNorm2d(ngf * 4)

            # Residual blocks
            self.res_blocks = nn.ModuleList([
                ResidualBlock.create(ngf * 4) for _ in range(n_blocks)
            ])

            # Upsampling
            self.deconv1 = UpsampleConvLayer.create(ngf * 4, ngf * 2, 3, 1, upsample=2)
            self.in4 = nn.InstanceNorm2d(ngf * 2)
            self.deconv2 = UpsampleConvLayer.create(ngf * 2, ngf, 3, 1, upsample=2)
            self.in5 = nn.InstanceNorm2d(ngf)

            # Output convolution
            self.deconv3 = ConvLayer.create(ngf, output_nc, 9, 1)

            self.relu = nn.ReLU()

        def forward(self, x):
            # Encode
            y = self.relu(self.in1(self.conv1(x)))
            y = self.relu(self.in2(self.conv2(y)))
            y = self.relu(self.in3(self.conv3(y)))

            # Transform
            for block in self.res_blocks:
                y = block(y)

            # Decode
            y = self.relu(self.in4(self.deconv1(y)))
            y = self.relu(self.in5(self.deconv2(y)))
            y = self.deconv3(y)

            return y

    return MSGNet()


@TransferRegistry.register("msgnet")
class MSGNetTransfer(AbstractTransfer):
    """
    MSG-Net neural style transfer.

    Uses a multi-style generative network that performs
    fast feedforward style transfer.

    Features:
    - Neural network-based
    - GPU accelerated (if available)
    - Single forward pass
    """

    def __init__(
        self,
        color_strength: float = 0.7,
        luminance_strength: float = 0.0
    ):
        super().__init__(color_strength, luminance_strength)

        self._model = None
        self._device = None
        self._reference_tensor = None
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
        return "msgnet"

    @property
    def method_name(self) -> str:
        return "MSG-Net"

    @property
    def method_type(self) -> str:
        return "neural"

    def is_available(self) -> bool:
        """Check if MSG-Net is available (disabled - replaced by AdaIN)."""
        # Disabled: Replaced by AdaIN which provides better results
        return False

    def _load_model(self) -> None:
        """Load MSG-Net model."""
        if self._model is not None:
            return

        torch, nn, F = _ensure_torch()

        # Get device
        from ..device_manager import get_device_manager
        dm = get_device_manager()
        self._device = dm.device

        # Create model
        self._model = create_msgnet_model()
        self._model.to(self._device)
        self._model.eval()

        # Try to load pretrained weights
        try:
            from ..model_manager import get_model_manager
            mm = get_model_manager()
            model_path = mm.get_model_path("msgnet")
            state_dict = torch.load(model_path, map_location=self._device)
            self._model.load_state_dict(state_dict, strict=False)
            logger.info("Loaded MSG-Net pretrained weights")
        except Exception as e:
            logger.warning(f"Could not load MSG-Net weights: {e}. Using random init.")

    def _preprocess_image(self, image: np.ndarray):
        """Convert BGR numpy array to torch tensor."""
        torch, _, _ = _ensure_torch()

        # BGR to RGB
        rgb = image[:, :, ::-1].copy()

        # HWC to CHW, normalize to [0, 1]
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor.to(self._device)

    def _postprocess_image(self, tensor) -> np.ndarray:
        """Convert torch tensor back to BGR numpy array."""
        # Remove batch dimension, move to CPU
        tensor = tensor.squeeze(0).cpu()

        # Clamp to [0, 1]
        tensor = tensor.clamp(0, 1)

        # CHW to HWC, denormalize
        image = (tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

        # RGB to BGR
        return image[:, :, ::-1].copy()

    def load_reference(self, image: np.ndarray) -> None:
        """
        Load reference image for style extraction.

        Args:
            image: BGR reference image
        """
        if not self._torch_available:
            raise RuntimeError("PyTorch is not available")

        self._load_model()
        self._reference_tensor = self._preprocess_image(image)
        self._reference_loaded = True

    def _apply_transfer(
        self,
        image: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """
        Apply MSG-Net style transfer.

        Args:
            image: BGR target image
            strength: Transfer strength

        Returns:
            Transferred BGR image
        """
        torch, _, _ = _ensure_torch()

        # Preprocess
        content = self._preprocess_image(image)

        # Forward pass
        with torch.no_grad():
            output = self._model(content)

        # Postprocess
        result = self._postprocess_image(output)

        return result
