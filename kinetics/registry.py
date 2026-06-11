# kinetics/registry.py
from __future__ import annotations

from typing import Dict, Type

from .base import LibraryBuilder

_REGISTRY: Dict[str, Type[LibraryBuilder]] = {}


def register(cls: Type[LibraryBuilder]) -> Type[LibraryBuilder]:
    """Class decorator to register gamma library builders by name."""
    if not getattr(cls, "name", None):
        raise ValueError("GammaLibraryBuilder subclasses must have a 'name' attribute.")

    key = cls.name.lower()
    if key in _REGISTRY:
        raise ValueError(f"Gamma library builder name '{cls.name}' is already registered.")

    _REGISTRY[key] = cls
    return cls


def create(name: str, config) -> LibraryBuilder:
    """
    Create a registered builder instance from a config object.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Gamma library builder '{name}' is not registered. "
            f"Available: {available()}"
        )

    builder_cls = _REGISTRY[key]
    return builder_cls(config)


def available() -> list[str]:
    return sorted(_REGISTRY.keys())