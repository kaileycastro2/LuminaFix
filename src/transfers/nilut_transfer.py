"""
NILUT Style Transfer - Neural Implicit 3D Lookup Tables.

Based on: "NILUT: Conditional Neural Implicit 3D Lookup Tables for Image Enhancement" (AAAI 2024)
Reference: https://github.com/mv-lab/nilut
"""

import time
from typing import Optional
from pathlib import Path
import numpy as np
import logging

from .abstract_transfer import AbstractTransfer
from .base import ProtectionMasks
from .registry import TransferRegistry

logger = logging.getLogger(__name__)

# Lazy imports for PyTorch
_torch = None
_nn = None

# Cache for loaded universal model state dicts (path -> state_dict)
# Avoids re-reading the .pt file from disk on every request
_universal_model_state_cache = {}


def _ensure_torch():
    """Lazily import PyTorch."""
    global _torch, _nn
    if _torch is None:
        import torch
        import torch.nn as nn
        _torch = torch
        _nn = nn
    return _torch, _nn


class NILUT_Model:
    """
    Neural Implicit 3D LUT Model.

    Simple MLP that learns a color transformation.
    Takes A,B (LAB) input, outputs transformed A,B.
    Operates in LAB space to preserve luminance.
    """

    @staticmethod
    def create(in_features=3, hidden_features=256, hidden_layers=3, out_features=3, res=True):
        """Create NILUT model architecture (per-reference model)."""
        torch, nn = _ensure_torch()

        class NILUT(nn.Module):
            def __init__(self):
                super().__init__()
                self.res = res

                # Build layers
                layers = []

                # Input layer
                layers.append(nn.Linear(in_features, hidden_features))
                layers.append(nn.ReLU(inplace=True))

                # Hidden layers
                for _ in range(hidden_layers):
                    layers.append(nn.Linear(hidden_features, hidden_features))
                    layers.append(nn.Tanh())

                # Output layer
                layers.append(nn.Linear(hidden_features, out_features))

                if not res:
                    layers.append(nn.Sigmoid())

                self.net = nn.Sequential(*layers)

            def forward(self, x):
                out = self.net(x)
                if self.res:
                    out = torch.clamp(out + x, 0, 1)
                return out

        return NILUT()

    @staticmethod
    def create_universal(hidden_features=256, hidden_layers=4):
        """
        Create universal NILUT model that accepts reference stats as conditioning.

        Input: [A, B, ref_A_mean, ref_A_std, ref_B_mean, ref_B_std] = 6 features
        Output: [A, B] = 2 features
        """
        torch, nn = _ensure_torch()

        class UniversalNILUT(nn.Module):
            def __init__(self):
                super().__init__()

                # Input: content A,B (2) + reference stats (4) = 6
                in_features = 6
                out_features = 2

                layers = []

                # Input layer
                layers.append(nn.Linear(in_features, hidden_features))
                layers.append(nn.ReLU(inplace=True))

                # Hidden layers
                for _ in range(hidden_layers):
                    layers.append(nn.Linear(hidden_features, hidden_features))
                    layers.append(nn.Tanh())

                # Output layer
                layers.append(nn.Linear(hidden_features, out_features))

                self.net = nn.Sequential(*layers)

            def forward(self, x):
                # x shape: [N, 6] where first 2 are content A,B
                content_ab = x[:, :2]
                out = self.net(x)
                # Residual connection on A,B only
                out = torch.clamp(out + content_ab, 0, 1)
                return out

        return UniversalNILUT()


@TransferRegistry.register("nilut")
class NILUTTransfer(AbstractTransfer):
    """
    NILUT neural color transfer in LAB space.

    Uses Neural Implicit 3D LUT for fast, consistent
    color transformation. Operates on A,B channels only,
    preserving original luminance (L).

    Features:
    - Color-only transfer (no texture changes)
    - Luminance preserved (like Reinhard)
    - Fast inference (simple MLP)
    - Learns from reference image statistics
    """

    # Max image dimension to prevent OOM
    MAX_IMAGE_SIZE = 2048

    # Universal model path
    UNIVERSAL_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "nilut" / "latest" / "universal.pt"

    def __init__(
        self,
        color_strength: float = 0.7,
        luminance_strength: float = 0.0,
        use_universal: bool = False
    ):
        super().__init__(color_strength, luminance_strength)

        self._model = None
        self._universal_model = None
        self._device = None
        self._reference_colors = None
        self._reference_stats = None  # For universal model: (A_mean, A_std, B_mean, B_std)
        self._reference_l_stats = None  # (L_mean, L_std) of reference, range 0-255
        self._use_universal = use_universal
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
        return "nilut"

    @property
    def method_name(self) -> str:
        return "NILUT"

    @property
    def method_type(self) -> str:
        return "neural"

    @property
    def neon_blend_factor(self) -> float:
        """Skip neon protection blending - we handle it in _apply_transfer instead."""
        return 0.0  # Disable post-transfer blending (neon already preserved)

    def _detect_neon_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Detect neon/highly saturated regions.

        Args:
            image: BGR image

        Returns:
            Mask (float32, 0-1) where 1 = neon region
        """
        import cv2

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        saturation = hsv[:, :, 1] / 255.0
        value = hsv[:, :, 2] / 255.0

        # Neon: medium-high saturation (>0.5) AND reasonably bright (>0.3)
        # Lowered thresholds to catch more vibrant colors
        neon_mask = ((saturation > 0.5) & (value > 0.3)).astype(np.float32)

        # Smooth edges with Gaussian blur for gradual transition
        neon_mask = cv2.GaussianBlur(neon_mask, (21, 21), 0)

        return neon_mask

    def _detect_skin_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Detect skin tone regions using YCbCr and HSV color spaces.

        Args:
            image: BGR image

        Returns:
            Mask (float32, 0-1) where 1 = skin region
        """
        import cv2

        # Convert to YCbCr color space (better for skin detection)
        ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        # Skin detection thresholds in YCbCr
        # Y: 0-255, Cb: 77-127, Cr: 133-173
        lower_ycbcr = np.array([0, 133, 77], dtype=np.uint8)
        upper_ycbcr = np.array([255, 173, 127], dtype=np.uint8)
        mask_ycbcr = cv2.inRange(ycbcr, lower_ycbcr, upper_ycbcr)

        # Also use HSV for additional filtering
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
        upper_hsv = np.array([20, 170, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Combine both masks
        skin_mask = cv2.bitwise_and(mask_ycbcr, mask_hsv).astype(np.float32) / 255.0

        # Clean up noise with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

        # Smooth edges for gradual blending
        skin_mask = cv2.GaussianBlur(skin_mask, (15, 15), 0)

        return skin_mask

    def _extract_luminance_details(self, image: np.ndarray) -> np.ndarray:
        """
        Extract high-frequency luminance details from image.

        Args:
            image: BGR image

        Returns:
            Detail layer (high-frequency information)
        """
        import cv2

        # Convert to LAB and extract luminance
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)

        # Apply bilateral filter to get smooth base layer
        base = cv2.bilateralFilter(l_channel.astype(np.uint8), 9, 75, 75).astype(np.float32)

        # Detail = original - base
        detail = l_channel - base

        return detail

    def _separate_highlights_shadows(self, image: np.ndarray) -> tuple:
        """
        Separate highlight and shadow regions.

        Args:
            image: BGR image

        Returns:
            Tuple of (highlight_mask, shadow_mask) each float32 0-1
        """
        import cv2

        # Convert to LAB and extract luminance
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32) / 255.0

        # Highlight mask: bright regions (L > 0.7)
        highlight_mask = np.clip((l_channel - 0.7) / 0.3, 0, 1)

        # Shadow mask: dark regions (L < 0.3)
        shadow_mask = np.clip((0.3 - l_channel) / 0.3, 0, 1)

        # Smooth for gradual transitions
        highlight_mask = cv2.GaussianBlur(highlight_mask, (15, 15), 0)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (15, 15), 0)

        return highlight_mask, shadow_mask

    def is_available(self) -> bool:
        """Check if PyTorch is available."""
        return self._torch_available

    def _get_device(self):
        """Get best available device."""
        torch, _ = _ensure_torch()
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def save_model(self, path: str) -> None:
        """
        Save trained model to file.

        Args:
            path: Path to save model (.pt file)
        """
        if self._model is None:
            raise RuntimeError("No model to save - train first")

        torch, _ = _ensure_torch()
        torch.save(self._model.state_dict(), path)
        logger.info(f"NILUT model saved to {path}")

    def _compute_reference_l_stats(self, image: np.ndarray) -> tuple:
        """
        Compute L channel mean and std (in 0-255 range) from reference image.

        Used to match the target's tonality to the reference via shift+scale,
        replacing the synthetic S-curve.
        """
        import cv2
        l = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        return float(l.mean()), float(l.std())

    def _compute_reference_stats(self, image: np.ndarray) -> tuple:
        """
        Compute A,B channel statistics from reference image.

        Returns:
            Tuple of (A_mean, A_std, B_mean, B_std) normalized to [0,1]
        """
        import cv2

        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0

        # Compute stats for A and B channels
        a_channel = lab[:, :, 1].flatten()
        b_channel = lab[:, :, 2].flatten()

        return (
            a_channel.mean(),
            a_channel.std(),
            b_channel.mean(),
            b_channel.std()
        )

    def save_universal_model(self, path: str = None) -> None:
        """Save trained universal model to file."""
        if self._universal_model is None:
            raise RuntimeError("No universal model to save - train first")

        torch, _ = _ensure_torch()
        save_path = path or str(self.UNIVERSAL_MODEL_PATH)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._universal_model.state_dict(), save_path)
        logger.info(f"Universal NILUT model saved to {save_path}")

    def load_universal_model(self, path: str = None) -> bool:
        """
        Load pre-trained universal model.

        Returns:
            True if loaded successfully, False otherwise
        """
        torch, _ = _ensure_torch()

        load_path = Path(path) if path else self.UNIVERSAL_MODEL_PATH
        if not load_path.exists():
            logger.warning(f"Universal model not found: {load_path}")
            return False

        self._device = self._get_device()
        self._universal_model = NILUT_Model.create_universal(
            hidden_features=256,
            hidden_layers=4
        ).to(self._device)

        cache_key = str(load_path)
        if cache_key in _universal_model_state_cache:
            state_dict = _universal_model_state_cache[cache_key]
            logger.info(f"Universal NILUT model loaded from cache: {load_path}")
        else:
            state_dict = torch.load(str(load_path), map_location=self._device, weights_only=True)
            _universal_model_state_cache[cache_key] = state_dict
            logger.info(f"Universal NILUT model loaded from disk: {load_path}")

        self._universal_model.load_state_dict(state_dict)
        self._universal_model.eval()
        return True

    @staticmethod
    def is_universal_model_available() -> bool:
        """Check if universal model exists."""
        return NILUTTransfer.UNIVERSAL_MODEL_PATH.exists()

    def train_universal_model(
        self,
        reference_images: list,
        sample_images: list,
        epochs: int = 500,
        save_path: str = None
    ) -> None:
        """
        Train universal NILUT model on multiple reference/content pairs.

        This model learns to transfer colors based on reference statistics,
        so it works with ANY reference without retraining.

        Args:
            reference_images: List of BGR reference images
            sample_images: List of BGR content/sample images
            epochs: Training epochs
            save_path: Optional path to save model
        """
        import cv2
        torch, nn = _ensure_torch()

        self._device = self._get_device()
        self._universal_model = NILUT_Model.create_universal(
            hidden_features=256,
            hidden_layers=4
        ).to(self._device)

        # Collect training pairs: (content_ab, ref_stats) -> target_ab
        all_inputs = []
        all_targets = []

        # Pre-sample content colors ONCE (avoids re-sampling per reference)
        logger.info("  Preparing content colors from %d images...", len(sample_images))
        content_samples_per_img = 5000
        all_content_ab = []
        for i, content_img in enumerate(sample_images):
            _, content_ab = self._sample_colors(content_img, n_samples=content_samples_per_img)
            all_content_ab.append(content_ab)
        content_ab_pool = np.concatenate(all_content_ab, axis=0)
        logger.info("  Content pool: %d color samples ready", len(content_ab_pool))

        # Loop over references only (fast: just len(reference_images) iterations)
        samples_per_ref = max(2000, 50000 // max(len(reference_images), 1))
        total_refs = len(reference_images)

        for idx, ref_img in enumerate(reference_images):
            if (idx + 1) % 10 == 0 or idx == 0:
                logger.info("  Processing reference %d/%d...", idx + 1, total_refs)

            ref_stats = self._compute_reference_stats(ref_img)
            _, ref_ab = self._sample_colors(ref_img, n_samples=samples_per_ref)

            # Sample from pre-computed content pool
            pool_indices = np.random.choice(len(content_ab_pool), samples_per_ref, replace=True)
            content_ab = content_ab_pool[pool_indices]

            content_matched, ref_matched = self._histogram_match_pairing(
                content_ab, ref_ab
            )

            n = len(content_matched)
            ref_stats_expanded = np.tile(ref_stats, (n, 1))
            inputs = np.concatenate([content_matched, ref_stats_expanded], axis=1)

            all_inputs.append(inputs)
            all_targets.append(ref_matched)

        # Combine all training data
        all_inputs = np.concatenate(all_inputs, axis=0).astype(np.float32)
        all_targets = np.concatenate(all_targets, axis=0).astype(np.float32)

        logger.info("  Total training samples: %d", len(all_inputs))
        logger.info(f"Training universal NILUT on {len(all_inputs)} samples")

        # Convert to tensors
        input_tensor = torch.FloatTensor(all_inputs).to(self._device)
        target_tensor = torch.FloatTensor(all_targets).to(self._device)

        # Training with adjusted scheduler for 1500 epochs
        optimizer = torch.optim.Adam(self._universal_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=450, gamma=0.5)
        criterion = nn.MSELoss()

        self._universal_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self._universal_model(input_tensor)
            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (epoch + 1) % 50 == 0:
                logger.info("  Epoch %d/%d, loss: %.6f", epoch + 1, epochs, loss.item())

        self._universal_model.eval()
        logger.info("Universal NILUT training complete, final loss: %.6f", loss.item())

        # Save model
        if save_path:
            self.save_universal_model(save_path)
        else:
            self.save_universal_model()

    def load_model(self, path: str) -> None:
        """
        Load pre-trained model from file.

        Args:
            path: Path to model file (.pt)
        """
        torch, _ = _ensure_torch()

        if not Path(path).exists():
            raise FileNotFoundError(f"Model not found: {path}")

        self._device = self._get_device()
        self._model = NILUT_Model.create(
            in_features=2,
            hidden_features=256,  # Updated to match new architecture
            hidden_layers=4,      # Updated to match new architecture
            out_features=2
        ).to(self._device)

        self._model.load_state_dict(torch.load(path, map_location=self._device, weights_only=True))
        self._model.eval()
        self._reference_loaded = True
        logger.info(f"NILUT model loaded from {path}")

    def pretrain_on_reference(self, reference_image: np.ndarray, sample_images: list, save_path: str = None) -> None:
        """
        Pre-train model on reference + multiple sample images.

        Use this offline to create a reusable model.
        Uses histogram matching for better color pairing (not L-sorting).

        Args:
            reference_image: BGR reference image
            sample_images: List of BGR sample content images
            save_path: Optional path to save model after training
        """
        import cv2

        # Collect A,B samples from reference (increased to 50K)
        _, style_ab = self._sample_colors(reference_image, n_samples=50000)

        # Collect A,B samples from all content images
        all_content_ab = []
        samples_per_image = 50000 // max(len(sample_images), 1)
        for img in sample_images:
            _, content_ab = self._sample_colors(img, n_samples=samples_per_image)
            all_content_ab.append(content_ab)

        content_ab_combined = np.concatenate(all_content_ab, axis=0)

        # Use histogram matching for pairing (instead of L-sorting)
        # Pair by color percentile rank in A and B channels separately
        content_matched, style_matched = self._histogram_match_pairing(
            content_ab_combined, style_ab
        )

        # Train with more epochs for better generalization (increased to 500)
        self._fit_lut(content_matched, style_matched, epochs=500)

        if save_path:
            self.save_model(save_path)

        logger.info(f"NILUT pre-trained on {len(sample_images)} sample images with histogram matching")

    def _histogram_match_pairing(self, content_ab: np.ndarray, style_ab: np.ndarray) -> tuple:
        """
        Create color pairs using histogram matching instead of L-sorting.

        Pairs colors by their percentile rank in A and B channels separately,
        ensuring red pairs with red, blue with blue (not by brightness).

        Args:
            content_ab: Content image A,B values [N, 2] normalized to [0,1]
            style_ab: Style/reference A,B values [M, 2] normalized to [0,1]

        Returns:
            Tuple of (content_matched, style_matched) arrays for training
        """
        # Match sizes
        n = min(len(content_ab), len(style_ab))

        # Sort both by A channel (green-red axis)
        content_a_sorted_idx = np.argsort(content_ab[:, 0])
        style_a_sorted_idx = np.argsort(style_ab[:, 0])

        # Sort both by B channel (blue-yellow axis)
        content_b_sorted_idx = np.argsort(content_ab[:, 1])
        style_b_sorted_idx = np.argsort(style_ab[:, 1])

        # Sample evenly from sorted indices
        content_a_indices = np.linspace(0, len(content_a_sorted_idx)-1, n//2, dtype=int)
        style_a_indices = np.linspace(0, len(style_a_sorted_idx)-1, n//2, dtype=int)
        content_b_indices = np.linspace(0, len(content_b_sorted_idx)-1, n//2, dtype=int)
        style_b_indices = np.linspace(0, len(style_b_sorted_idx)-1, n//2, dtype=int)

        # Get paired samples from A-channel sorting
        content_from_a = content_ab[content_a_sorted_idx[content_a_indices]]
        style_from_a = style_ab[style_a_sorted_idx[style_a_indices]]

        # Get paired samples from B-channel sorting
        content_from_b = content_ab[content_b_sorted_idx[content_b_indices]]
        style_from_b = style_ab[style_b_sorted_idx[style_b_indices]]

        # Combine both sets of pairs
        content_matched = np.concatenate([content_from_a, content_from_b], axis=0)
        style_matched = np.concatenate([style_from_a, style_from_b], axis=0)

        # Shuffle to mix A and B sorted pairs
        shuffle_idx = np.random.permutation(len(content_matched))
        content_matched = content_matched[shuffle_idx]
        style_matched = style_matched[shuffle_idx]

        logger.debug(f"Histogram match pairing: {len(content_matched)} pairs created")
        return content_matched, style_matched

    def _fit_lut(self, source_colors: np.ndarray, target_colors: np.ndarray, epochs: int = 500):
        """
        Fit NILUT model to map source A,B to target A,B colors.

        Args:
            source_colors: Source image A,B channels [N, 2] normalized to [0,1]
            target_colors: Target (reference) A,B channels [N, 2] normalized to [0,1]
            epochs: Training epochs (default increased to 500)
        """
        torch, nn = _ensure_torch()

        self._device = self._get_device()
        self._model = NILUT_Model.create(
            in_features=2,        # A,B channels only
            hidden_features=256,  # Increased from 128
            hidden_layers=4,      # Increased from 2
            out_features=2        # A,B channels only
        ).to(self._device)

        # Prepare data
        source_tensor = torch.FloatTensor(source_colors).to(self._device)
        target_tensor = torch.FloatTensor(target_colors).to(self._device)

        # Training with learning rate scheduler for better convergence
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=450, gamma=0.5)
        criterion = nn.MSELoss()

        self._model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self._model(source_tensor)
            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (epoch + 1) % 50 == 0:
                logger.info("  Epoch %d/%d, loss: %.6f", epoch + 1, epochs, loss.item())

        self._model.eval()
        logger.info("Training complete, final loss: %.6f", loss.item())

    def _sample_colors(self, image: np.ndarray, n_samples: int = 10000) -> tuple:
        """
        Sample random colors from image in LAB space.

        Returns:
            Tuple of (L channel [N, 1], A,B channels [N, 2]) normalized to [0,1]
        """
        import cv2

        h, w = image.shape[:2]
        total_pixels = h * w

        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Flatten to [N, 3]
        flat = lab.reshape(-1, 3) / 255.0  # Normalize to 0-1

        # Random sample
        if total_pixels > n_samples:
            indices = np.random.choice(total_pixels, n_samples, replace=False)
            flat = flat[indices]

        # Return L separately (for sorting) and A,B (for training)
        l_channel = flat[:, 0:1]  # [N, 1]
        ab_channels = flat[:, 1:3]  # [N, 2]

        return l_channel, ab_channels

    def _create_color_mapping(self, content_img: np.ndarray, style_img: np.ndarray):
        """
        Create color mapping by matching histograms and fitting NILUT.

        Uses histogram matching for pairing (not L-sorting), trains on A,B channels only.
        """
        # Sample colors from both images in LAB space (increased to 50K)
        _, content_ab = self._sample_colors(content_img, n_samples=50000)
        _, style_ab = self._sample_colors(style_img, n_samples=50000)

        # Use histogram matching for better color pairing
        content_matched, style_matched = self._histogram_match_pairing(content_ab, style_ab)

        # Fit NILUT to learn A,B transformation (increased epochs to 500)
        self._fit_lut(content_matched, style_matched, epochs=500)

    def load_reference(self, image: np.ndarray, model_path: Optional[str] = None, use_universal: bool = None) -> None:
        """
        Load reference image for style extraction.

        Args:
            image: BGR reference image
            model_path: Optional path to pre-trained per-reference model
            use_universal: If True, use universal model; if None, auto-detect
        """
        if not self._torch_available:
            raise RuntimeError("PyTorch is not available")

        # Cache reference L stats for tone matching (used by post-NILUT enhancements)
        self._reference_l_stats = self._compute_reference_l_stats(image)

        # Determine which mode to use
        if use_universal is None:
            use_universal = self._use_universal

        # Universal model mode - just compute reference stats
        if use_universal:
            if self._universal_model is None:
                if not self.load_universal_model():
                    logger.warning("Universal model not found, falling back to per-reference mode")
                    use_universal = False

        if use_universal and self._universal_model is not None:
            # Just store reference stats for universal model
            self._reference_stats = self._compute_reference_stats(image)
            self._reference_loaded = True
            self._use_universal = True
            logger.info(f"NILUT using universal model with reference stats: {self._reference_stats}")
            return

        # Per-reference model mode
        self._use_universal = False

        # If pre-trained model provided, load it directly (fast path)
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
            logger.info(f"NILUT using pre-trained model: {model_path}")
            return

        # Store reference for later fitting (on-the-fly training)
        # Resize if too large
        h, w = image.shape[:2]
        if max(h, w) > self.MAX_IMAGE_SIZE:
            import cv2
            scale = self.MAX_IMAGE_SIZE / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))

        self._reference_image = image.copy()
        self._reference_loaded = True
        logger.debug(f"NILUT reference loaded: {image.shape}")

    # LUT size for fast inference (33^2 = 1089 points vs 667K pixels)
    LUT_SIZE = 33

    def _build_ab_lut(self) -> np.ndarray:
        """
        Build a 2D LUT for A,B channel transformation.

        Creates a grid of A,B values, runs inference once, and returns
        a lookup table for fast application to any image.

        Returns:
            LUT array of shape [LUT_SIZE, LUT_SIZE, 2] mapping input A,B to output A,B
        """
        torch, _ = _ensure_torch()

        # Create grid of A,B values (normalized 0-1)
        a_vals = np.linspace(0, 1, self.LUT_SIZE)
        b_vals = np.linspace(0, 1, self.LUT_SIZE)

        # Create meshgrid
        a_grid, b_grid = np.meshgrid(a_vals, b_vals, indexing='ij')
        ab_grid = np.stack([a_grid, b_grid], axis=-1).reshape(-1, 2).astype(np.float32)

        # Run inference on grid (only 1089 points!)
        with torch.no_grad():
            input_tensor = torch.FloatTensor(ab_grid).to(self._device)
            output_tensor = self._model(input_tensor)
            ab_output = output_tensor.cpu().numpy()

        # Reshape to LUT and smooth to prevent color banding
        lut = ab_output.reshape(self.LUT_SIZE, self.LUT_SIZE, 2)
        lut = self._smooth_lut(lut)
        return lut

    def _apply_ab_lut(self, ab_channels: np.ndarray, lut: np.ndarray) -> np.ndarray:
        """
        Apply pre-computed A,B LUT using cv2.remap.

        Mathematically equivalent to the previous hand-rolled numpy bilinear
        interpolation (cv2.remap uses the same INTER_LINEAR weights), but
        runs in optimized C++ — typically 5-10x faster on large images.

        Args:
            ab_channels: A,B channels [H, W, 2] in range 0-255
            lut: LUT array [LUT_SIZE, LUT_SIZE, 2] normalized 0-1, indexed [a, b]

        Returns:
            Transformed A,B channels [H, W, 2] in range 0-255
        """
        import cv2

        # Scale 0-255 input values into LUT index space [0, LUT_SIZE-1].
        scale = (self.LUT_SIZE - 1) / 255.0
        # cv2.remap expects map_x (column) and map_y (row).
        # In `lut[a, b]`, a is the row axis (axis 0), b is the column axis (axis 1).
        map_y = ab_channels[:, :, 0].astype(np.float32) * scale  # a -> row
        map_x = ab_channels[:, :, 1].astype(np.float32) * scale  # b -> column

        out = cv2.remap(
            lut.astype(np.float32),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return (out * 255).clip(0, 255)

    def _build_universal_ab_lut(self) -> np.ndarray:
        """
        Build a 2D LUT using the universal model with current reference stats.

        Returns:
            LUT array of shape [LUT_SIZE, LUT_SIZE, 2] mapping input A,B to output A,B
        """
        torch, _ = _ensure_torch()

        if self._reference_stats is None:
            raise RuntimeError("Reference stats not computed")

        # Create grid of A,B values (normalized 0-1)
        a_vals = np.linspace(0, 1, self.LUT_SIZE)
        b_vals = np.linspace(0, 1, self.LUT_SIZE)

        # Create meshgrid
        a_grid, b_grid = np.meshgrid(a_vals, b_vals, indexing='ij')
        ab_grid = np.stack([a_grid, b_grid], axis=-1).reshape(-1, 2).astype(np.float32)

        # Append reference stats to each point
        n_points = ab_grid.shape[0]
        ref_stats = np.array(self._reference_stats, dtype=np.float32)
        ref_stats_expanded = np.tile(ref_stats, (n_points, 1))
        input_grid = np.concatenate([ab_grid, ref_stats_expanded], axis=1)

        # Run inference on grid
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_grid).to(self._device)
            output_tensor = self._universal_model(input_tensor)
            ab_output = output_tensor.cpu().numpy()

        # Reshape to LUT and smooth to prevent color banding
        lut = ab_output.reshape(self.LUT_SIZE, self.LUT_SIZE, 2)
        lut = self._smooth_lut(lut)
        return lut

    @staticmethod
    def _smooth_lut(lut: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Smooth the LUT to prevent color banding from discontinuities
        between adjacent grid points.

        Args:
            lut: LUT array [N, N, 2]
            kernel_size: Gaussian blur kernel size (odd number)

        Returns:
            Smoothed LUT array [N, N, 2]
        """
        import cv2
        smoothed = np.empty_like(lut)
        for c in range(lut.shape[2]):
            smoothed[:, :, c] = cv2.GaussianBlur(
                lut[:, :, c].astype(np.float32),
                (kernel_size, kernel_size), 0
            )
        return smoothed

    @staticmethod
    def _guided_filter_lab(guide: np.ndarray, src: np.ndarray, radius: int = 10, eps: float = 8000, subsample: int = 4) -> np.ndarray:
        """
        Fast guided filter applied only to A,B color channels in LAB space.

        Smooths color patches/inconsistencies from LUT mapping while
        preserving luminance detail (sharpness, texture) completely.
        Uses original image's L channel as structure guide.

        Args:
            guide: Original image (structure reference), BGR uint8
            src: Transferred image (to smooth), BGR uint8
            radius: Filter window radius
            eps: Regularization (higher = smoother in flat areas)
            subsample: Downscale factor for fast computation

        Returns:
            Smoothed BGR uint8 image (luminance unchanged)
        """
        import cv2

        h, w = guide.shape[:2]
        s = subsample
        r = max(1, radius // s)
        ksize = (2 * r + 1, 2 * r + 1)

        # Convert to LAB
        guide_lab = cv2.cvtColor(guide, cv2.COLOR_BGR2LAB)
        src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)

        # Keep src luminance untouched
        result_lab = src_lab.copy()

        # Use guide's L channel as structure reference
        guide_L = guide_lab[:, :, 0]

        # Downsample guide L for fast computation
        guide_L_s = cv2.resize(guide_L, (w // s, h // s), interpolation=cv2.INTER_AREA).astype(np.float32)

        # Filter only A and B channels (indices 1, 2)
        for c in [1, 2]:
            src_ch_s = cv2.resize(src_lab[:, :, c], (w // s, h // s), interpolation=cv2.INTER_AREA).astype(np.float32)

            mean_I = cv2.blur(guide_L_s, ksize)
            mean_p = cv2.blur(src_ch_s, ksize)
            corr_Ip = cv2.blur(guide_L_s * src_ch_s, ksize)
            corr_II = cv2.blur(guide_L_s * guide_L_s, ksize)

            var_I = corr_II - mean_I * mean_I
            cov_Ip = corr_Ip - mean_I * mean_p

            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I

            mean_a = cv2.blur(a, ksize)
            mean_b = cv2.blur(b, ksize)

            # Upsample coefficients to full resolution
            mean_a_full = cv2.resize(mean_a, (w, h), interpolation=cv2.INTER_LINEAR)
            mean_b_full = cv2.resize(mean_b, (w, h), interpolation=cv2.INTER_LINEAR)

            # Apply at full resolution using guide L
            q = mean_a_full * guide_L.astype(np.float32) + mean_b_full
            result_lab[:, :, c] = np.clip(q, 0, 255).astype(np.uint8)

        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    def apply_clahe_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast.

        CLAHE is applied ONLY to skin regions for targeted enhancement.

        Args:
            image: BGR image (NILUT result without CLAHE)

        Returns:
            BGR image with CLAHE applied to skin regions only
        """
        import cv2

        # Detect skin regions
        skin_mask = self._detect_skin_mask(image)

        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Extract L channel
        l_original = lab[:, :, 0].copy()

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_original.astype(np.uint8)).astype(np.float32)

        # Blend 50/50 ONLY on skin regions
        l_final = l_original * (1.0 - skin_mask * 0.5) + l_enhanced * (skin_mask * 0.5)

        # Replace L channel
        lab[:, :, 0] = l_final

        # Convert back to BGR
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        return result

    def apply_tonecurve_enhancement(self, image: np.ndarray, curve_strength: float = 0.5) -> np.ndarray:
        """
        Match the target's L (luminance) statistics to the reference's
        (Reinhard-style on L): shift to reference mean, scale to reference std.

        Skin regions are protected.

        Args:
            image: BGR image (NILUT result)
            curve_strength: Blend factor for the matched L (0-1).
                            0 = original L, 1 = fully matched.

        Returns:
            BGR image with L-stat matching applied (excluding skin)
        """
        import cv2

        if self._reference_l_stats is None or curve_strength <= 0:
            return image

        # Detect skin regions
        skin_mask = self._detect_skin_mask(image)
        skin_mask_3d = skin_mask[:, :, np.newaxis]

        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_channel = lab[:, :, 0]

        # Match L stats to reference: (L - tgt_mean) * (ref_std / tgt_std) + ref_mean
        ref_mean, ref_std = self._reference_l_stats
        tgt_mean = float(l_channel.mean())
        tgt_std = float(l_channel.std())
        if tgt_std < 1e-3:
            return image
        l_matched = (l_channel - tgt_mean) * (ref_std / tgt_std) + ref_mean

        # Blend by strength
        l_final = l_channel * (1.0 - curve_strength) + l_matched * curve_strength
        lab[:, :, 0] = np.clip(l_final, 0, 255)

        enhanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # Skin protection: keep original on skin
        result = enhanced.astype(np.float32) * (1.0 - skin_mask_3d) + image.astype(np.float32) * skin_mask_3d
        return result.astype(np.uint8)

    def apply_tonecurve_saturation_enhancement(self, image: np.ndarray, curve_strength: float = 0.5, saturation_boost: float = 1.4) -> np.ndarray:
        """
        L-stat match (Reinhard on L) to transfer the reference's tonality,
        plus chroma boost on A,B for color separation.

        Skin regions are protected.

        Args:
            image: BGR image (NILUT result)
            curve_strength: Blend factor for L-stat matching (0-1).
                            0 = original L, 1 = fully matched.
            saturation_boost: Chroma multiplier (1.4 = 40% boost)

        Returns:
            BGR image with L-stat matching + saturation applied (excluding skin)
        """
        import cv2

        # Detect skin regions
        skin_mask = self._detect_skin_mask(image)
        skin_mask_3d = skin_mask[:, :, np.newaxis]

        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_channel = lab[:, :, 0]

        # L-stat matching (only if we have reference stats and strength > 0)
        if self._reference_l_stats is not None and curve_strength > 0:
            ref_mean, ref_std = self._reference_l_stats
            tgt_mean = float(l_channel.mean())
            tgt_std = float(l_channel.std())
            if tgt_std >= 1e-3:
                l_matched = (l_channel - tgt_mean) * (ref_std / tgt_std) + ref_mean
                l_final = l_channel * (1.0 - curve_strength) + l_matched * curve_strength
                lab[:, :, 0] = np.clip(l_final, 0, 255)

        # Saturation boost on A,B
        ab_channels = lab[:, :, 1:3]
        ab_mean = np.array([128, 128], dtype=np.float32).reshape(1, 1, 2)
        ab_boosted = ab_mean + (ab_channels - ab_mean) * saturation_boost
        lab[:, :, 1:3] = np.clip(ab_boosted, 0, 255)

        enhanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # Skin protection
        result = enhanced.astype(np.float32) * (1.0 - skin_mask_3d) + image.astype(np.float32) * skin_mask_3d
        return result.astype(np.uint8)

    def apply_chroma_boost(self, image: np.ndarray, boost_strength: float = 1.35) -> np.ndarray:
        """
        Apply increased chroma boost for more vibrant, saturated colors.

        This amplifies the color saturation of NILUT results by pushing
        A,B channels further from neutral gray.

        Args:
            image: BGR image (NILUT result)
            boost_strength: Chroma multiplier (1.35 = 35% boost, default)

        Returns:
            BGR image with enhanced color saturation
        """
        import cv2

        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Apply chroma boost to A,B channels
        ab_channels = lab[:, :, 1:3]
        ab_mean = np.array([128, 128], dtype=np.float32).reshape(1, 1, 2)
        ab_boosted = ab_mean + (ab_channels - ab_mean) * boost_strength
        lab[:, :, 1:3] = np.clip(ab_boosted, 0, 255)

        # Convert back to BGR
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        return result

    def transfer_with_strength_map(
        self,
        target: np.ndarray,
        strength_map: np.ndarray,
        masks: Optional[ProtectionMasks] = None,
    ) -> np.ndarray:
        """
        Apply NILUT with a per-pixel strength map (segment-aware tuning).

        Skips the scalar-strength postprocess blend in AbstractTransfer because
        per-pixel blending already happens inside _apply_transfer.

        Args:
            target: BGR target image (uint8).
            strength_map: HxW float32 array; each pixel's NILUT blend weight in [0, ~1.5].
            masks: optional ProtectionMasks (skin/neon/lips/eyes).

        Returns:
            BGR uint8 result.
        """
        import cv2

        if not self._reference_loaded:
            raise RuntimeError(f"{self.method_name}: Reference not loaded.")

        if strength_map.shape[:2] != target.shape[:2]:
            strength_map = cv2.resize(
                strength_map,
                (target.shape[1], target.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        original = target.copy()
        preprocessed = self._preprocess(target)
        result = self._apply_transfer(preprocessed, strength_map)

        if masks is not None:
            result = self._apply_protection(original, result, masks)

        return result

    def _apply_transfer(
        self,
        image: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """
        Apply NILUT color transfer in LAB space with post-processing enhancements.

        Milestone 2 enhancements (base NILUT):
        - Detail preservation from original
        - Skin tone preservation
        - Better color separation and depth

        Note: This processes WITHOUT CLAHE contrast enhancement.
        Use apply_clahe_enhancement() separately for adaptive contrast.

        Args:
            image: BGR target image
            strength: Transfer strength

        Returns:
            Color-transferred BGR image (NILUT only, no CLAHE)
        """
        import cv2
        torch, _ = _ensure_torch()

        original_h, original_w = image.shape[:2]
        total_t0 = time.perf_counter()
        logger.info("NILUT timing [start]: image %dx%d (%.1fM pixels)",
                    original_w, original_h, (original_w * original_h) / 1e6)

        # ===== PRE-PROCESSING =====
        # Extract luminance details from original
        t0 = time.perf_counter()
        original_detail = self._extract_luminance_details(image)
        logger.info("NILUT timing [extract_luminance_details]: %.0f ms",
                    (time.perf_counter() - t0) * 1000)

        # Detect skin regions
        t0 = time.perf_counter()
        skin_mask = self._detect_skin_mask(image)
        skin_mask_3d = skin_mask[:, :, np.newaxis]
        logger.info("NILUT timing [detect_skin_mask]: %.0f ms",
                    (time.perf_counter() - t0) * 1000)

        # Detect neon regions
        t0 = time.perf_counter()
        neon_mask = self._detect_neon_mask(image)
        neon_mask_3d = neon_mask[:, :, np.newaxis]
        logger.info("NILUT timing [detect_neon_mask]: %.0f ms",
                    (time.perf_counter() - t0) * 1000)

        # Separate highlights and shadows for detail preservation
        t0 = time.perf_counter()
        highlight_mask, shadow_mask = self._separate_highlights_shadows(image)
        logger.info("NILUT timing [separate_highlights_shadows]: %.0f ms",
                    (time.perf_counter() - t0) * 1000)

        # ===== BUILD/LOAD LUT =====
        t0 = time.perf_counter()
        # Use universal model if available and enabled
        if self._use_universal and self._universal_model is not None:
            # Build LUT using universal model with reference stats
            lut = self._build_universal_ab_lut()
        else:
            # Per-reference model mode
            # Resize if needed for training
            if max(original_h, original_w) > self.MAX_IMAGE_SIZE:
                scale = self.MAX_IMAGE_SIZE / max(original_h, original_w)
                image_small = cv2.resize(image, (int(original_w * scale), int(original_h * scale)))
            else:
                image_small = image

            # Fit NILUT on this image pair (trains on A,B channels)
            # Skip if model already pre-loaded
            if self._model is None:
                self._create_color_mapping(image_small, self._reference_image)

            # Build LUT once (fast: only 1089 inference points)
            lut = self._build_ab_lut()
        logger.info("NILUT timing [build_lut]: %.0f ms",
                    (time.perf_counter() - t0) * 1000)

        # ===== APPLY NILUT TRANSFORMATION =====
        # Convert full image to LAB
        t0 = time.perf_counter()
        h, w = image.shape[:2]
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_original = lab.copy()

        # Extract channels
        l_channel = lab[:, :, 0:1]  # Keep original L (luminance)
        ab_channels = lab[:, :, 1:3]  # A,B to transform
        ab_original = ab_channels.copy()  # Save original A,B for neon areas
        logger.info("NILUT timing [bgr2lab + copy]: %.0f ms",
                    (time.perf_counter() - t0) * 1000)

        # Apply LUT (fast: uses pre-computed lookup)
        t0 = time.perf_counter()
        ab_transformed = self._apply_ab_lut(ab_channels, lut)
        logger.info("NILUT timing [apply_ab_lut]: %.0f ms",
                    (time.perf_counter() - t0) * 1000)

        # Blend A,B based on strength (allows extrapolation beyond 1.0)
        # `strength` may be a scalar OR an HxW per-pixel map (segment-aware tuning).
        t0 = time.perf_counter()
        if isinstance(strength, np.ndarray):
            s_map = strength.astype(np.float32)
            if s_map.ndim == 2:
                s_map = s_map[:, :, np.newaxis]
            ab_transformed = ab_channels * (1.0 - s_map) + ab_transformed * s_map
        else:
            ab_transformed = ab_channels * (1.0 - strength) + ab_transformed * strength

        # SKIP transformation on neon areas - keep original A,B
        ab_final = ab_transformed * (1.0 - neon_mask_3d) + ab_original * neon_mask_3d

        # ===== POST-PROCESSING: DETAIL RESTORATION =====
        # NOTE: CLAHE is NOT applied here in base NILUT
        # Use apply_clahe_enhancement() separately for contrast enhancement
        # Restore high-frequency details from original
        # Add back 30% of original detail to preserve texture
        l_channel[:, :, 0] = np.clip(l_channel[:, :, 0] + original_detail * 0.3, 0, 255)

        # ===== POST-PROCESSING: COLOR SEPARATION & DEPTH =====
        # Enhance chroma (color separation) selectively
        # Increase saturation in A,B channels by 10% for better color separation
        ab_mean = np.array([128, 128], dtype=np.float32).reshape(1, 1, 2)
        ab_enhanced = ab_mean + (ab_final - ab_mean) * 1.1
        ab_final = np.clip(ab_enhanced, 0, 255)

        # Merge enhanced L with final A,B
        lab_result = np.concatenate([l_channel, ab_final], axis=2).astype(np.uint8)
        logger.info("NILUT timing [strength_blend + neon_blend + detail + chroma]: %.0f ms",
                    (time.perf_counter() - t0) * 1000)

        # Convert back to BGR
        t0 = time.perf_counter()
        result = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)
        logger.info("NILUT timing [lab2bgr]: %.0f ms",
                    (time.perf_counter() - t0) * 1000)

        # ===== POST-PROCESSING: SKIN TONE PRESERVATION =====
        # Blend original skin tones back at 70% weight
        # This preserves natural skin color while allowing some style transfer
        t0 = time.perf_counter()
        result = result.astype(np.float32)
        image_float = image.astype(np.float32)
        result = result * (1.0 - skin_mask_3d * 0.7) + image_float * (skin_mask_3d * 0.7)
        result = result.astype(np.uint8)
        logger.info("NILUT timing [skin_blend_final]: %.0f ms",
                    (time.perf_counter() - t0) * 1000)

        logger.info("NILUT timing [TOTAL _apply_transfer]: %.0f ms",
                    (time.perf_counter() - total_t0) * 1000)

        return result
