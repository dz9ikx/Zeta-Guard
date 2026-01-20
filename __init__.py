"""
Zeta-Guard: AI Training Stability Guardian
"""


try:
    from .guard import ZetaGuard, StabilityResult
except ImportError:

    from guard import ZetaGuard, StabilityResult

__version__ = "0.0.1"
__author__ = "dz9ikx"
__all__ = ["ZetaGuard", "StabilityResult"]
