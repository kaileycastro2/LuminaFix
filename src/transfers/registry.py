"""
Transfer Registry - Factory pattern for style transfer methods.

Implements the Open/Closed principle: new methods can be added
without modifying existing code by using the @register decorator.
"""

from typing import Dict, Type, List, Optional, Any
import logging

from .base import StyleTransferProtocol

logger = logging.getLogger(__name__)


class TransferRegistry:
    """
    Registry for style transfer methods.

    Uses decorator pattern for registration, supporting
    the Open/Closed principle.

    Usage:
        @TransferRegistry.register("my_method")
        class MyTransfer(AbstractTransfer):
            ...

        # Get instance
        method = TransferRegistry.get("my_method", color_strength=0.8)

        # List available
        methods = TransferRegistry.get_all_available()
    """

    _methods: Dict[str, Type[StyleTransferProtocol]] = {}
    _instances: Dict[str, StyleTransferProtocol] = {}

    @classmethod
    def register(cls, method_id: str):
        """
        Decorator to register a transfer method.

        Args:
            method_id: Unique identifier for the method

        Usage:
            @TransferRegistry.register("reinhard")
            class ReinhardTransfer(AbstractTransfer):
                ...
        """
        def decorator(klass: Type[StyleTransferProtocol]) -> Type[StyleTransferProtocol]:
            if method_id in cls._methods:
                logger.warning(f"Overwriting existing method: {method_id}")
            cls._methods[method_id] = klass
            logger.debug(f"Registered transfer method: {method_id}")
            return klass
        return decorator

    @classmethod
    def get(
        cls,
        method_id: str,
        **kwargs
    ) -> StyleTransferProtocol:
        """
        Get or create a transfer method instance.

        Args:
            method_id: Method identifier
            **kwargs: Arguments passed to method constructor

        Returns:
            Style transfer method instance

        Raises:
            ValueError: If method_id is unknown
        """
        if method_id not in cls._methods:
            available = list(cls._methods.keys())
            raise ValueError(
                f"Unknown method: {method_id}. Available: {available}"
            )

        # Create new instance with provided kwargs
        return cls._methods[method_id](**kwargs)

    @classmethod
    def get_cached(
        cls,
        method_id: str,
        **kwargs
    ) -> StyleTransferProtocol:
        """
        Get a cached instance (singleton per method_id).

        Args:
            method_id: Method identifier
            **kwargs: Arguments for first instantiation only

        Returns:
            Cached style transfer method instance
        """
        if method_id not in cls._instances:
            cls._instances[method_id] = cls.get(method_id, **kwargs)
        return cls._instances[method_id]

    @classmethod
    def get_all_available(
        cls,
        **kwargs
    ) -> List[StyleTransferProtocol]:
        """
        Get instances of all available methods.

        Only returns methods where is_available() returns True.

        Args:
            **kwargs: Arguments passed to each method constructor

        Returns:
            List of available method instances
        """
        available = []
        for method_id in cls._methods:
            try:
                instance = cls.get(method_id, **kwargs)
                if instance.is_available():
                    available.append(instance)
                else:
                    logger.debug(f"Method {method_id} not available")
            except Exception as e:
                logger.warning(f"Failed to instantiate {method_id}: {e}")
        return available

    @classmethod
    def list_methods(cls) -> List[Dict[str, Any]]:
        """
        List all registered methods with metadata.

        Returns:
            List of method metadata dicts
        """
        result = []
        for method_id, method_class in cls._methods.items():
            try:
                instance = cls.get(method_id)
                result.append({
                    "id": method_id,
                    "name": instance.method_name,
                    "type": instance.method_type,
                    "available": instance.is_available()
                })
            except Exception as e:
                result.append({
                    "id": method_id,
                    "name": method_id,
                    "type": "unknown",
                    "available": False,
                    "error": str(e)
                })
        return result

    @classmethod
    def is_registered(cls, method_id: str) -> bool:
        """Check if a method is registered."""
        return method_id in cls._methods

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached instances."""
        cls._instances.clear()

    @classmethod
    def clear_all(cls) -> None:
        """Clear all registrations and cache (mainly for testing)."""
        cls._methods.clear()
        cls._instances.clear()


# Module-level singleton accessor
_registry: Optional[TransferRegistry] = None


def get_registry() -> Type[TransferRegistry]:
    """Get the TransferRegistry class (it uses class methods)."""
    return TransferRegistry
