#kinetics.registry.py
from __future__ import annotations
from typing import Dict, Type
from .base import GammaLibraryBuilder, GammaLibraryResult

_REGISTRY:Dict[str, Type[GammaLibraryBuilder]] = {}
def register(cls: Type[GammaLibraryBuilder]) -> type[GammaLibraryBuilder]:
    """Class decorator to register gamma library builders by name."""
    if not getattr(cls, 'name', None):
        raise ValueError("GammaLibraryBuilder subclasses must have a 'name' attribute.")
    key = cls.name.lower()
    if key in _REGISTRY:
        raise ValueError(f"Gamma library builder name '{cls.name}' is already registered.")
    _REGISTRY[key] = cls
    return cls
def create(name:str, **params) -> GammaLibraryResult:
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Gamma library builder '{name}' is not registered. Available: {available()}")
    builder = _REGISTRY[key](**params)
    return builder.build_library(**params)
def available():
    return sorted(_REGISTRY.keys())
