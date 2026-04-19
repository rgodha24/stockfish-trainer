"""ranger22 - Optimized Ranger21

- ``ranger22.Ranger22`` is the optimized version. It is a drop-in replacement
  with the same constructor signature and (within FP32 tolerance) the same
  update semantics.
"""

from .optimizer import Ranger22  # noqa: F401

__all__ = ["Ranger22"]
