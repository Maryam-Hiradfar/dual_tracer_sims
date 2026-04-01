# separation/registry.py
from __future__ import annotations

from typing import Dict, Type

from .base import SeparationAlgorithm


_REGISTRY: Dict[str, Type[SeparationAlgorithm]] = {}


def register(cls: Type[SeparationAlgorithm]) -> Type[SeparationAlgorithm]:
    """Class decorator to register separation algorithms by name."""
    if not getattr(cls, "name", None):
        raise ValueError("SeparationAlgorithm subclasses must have a 'name' attribute.")

    key = cls.name.lower()
    if key in _REGISTRY:
        raise ValueError(f"Algorithm name '{cls.name}' is already registered.")

    _REGISTRY[key] = cls
    return cls


def create(name: str, config) -> SeparationAlgorithm:
    """
    Create a registered separation algorithm instance from a config object.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Separation algorithm '{name}' is not registered. "
            f"Available: {available()}"
        )

    algorithm_cls = _REGISTRY[key]
    return algorithm_cls(config)


def available() -> list[str]:
    return sorted(_REGISTRY.keys())