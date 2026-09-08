"""NOT rembg. Same two-function surface, different backend.

Why: upstream calls rembg only to obtain a foreground alpha --

    session = rembg.new_session("u2net")
    rgba    = rembg.remove(pil_image, session=session)
    alpha   = np.array(rgba)[:, :, 3]

-- in trellis/pipelines/trellis_image_to_3d.py and voxhammer/edit_pipeline.py.
Nothing else about rembg is used.

Real rembg pulls onnxruntime and numba, whose Windows wheels drag in a native
stack that conflicts with the pip-built torch this environment needs. The
workspace already has an approved segmenter (rf-detr, segmentation head, with
CC BY 4.0 clothing instance data in 6-datasource/rf-detr-segmentation-data),
and BRIA RMBG is blocklisted, so rembg has no place here.

This module provides the alpha instead. Register a provider before use:

    import rembg
    rembg.set_mask_provider(fn)   # fn(PIL.Image) -> HxW uint8 alpha

With no provider registered it RAISES rather than returning a full-opaque
alpha. A permissive default would silently treat the whole frame as
foreground, and the caller's bbox crop would look like it worked.
"""

from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

__all__ = ["new_session", "remove", "set_mask_provider", "RembgShimError"]
__is_shim__ = True

_provider: Optional[Callable[[Image.Image], np.ndarray]] = None


class RembgShimError(RuntimeError):
    """Raised when an alpha is requested but no provider is registered."""


def set_mask_provider(fn: Optional[Callable[[Image.Image], np.ndarray]]) -> None:
    """Register the backend that turns an image into an HxW uint8 alpha."""
    global _provider
    _provider = fn


def new_session(model_name: str = "u2net", *args: Any, **kwargs: Any) -> str:
    """Accepted and ignored -- the provider decides the backend."""
    return f"shim:{model_name}"


def remove(data: Image.Image, *args: Any, **kwargs: Any) -> Image.Image:
    """Return ``data`` as RGBA with alpha from the registered provider."""
    if _provider is None:
        raise RembgShimError(
            "rembg.remove() called with no mask provider registered. This is "
            "the rembg shim; call rembg.set_mask_provider(fn) first, where "
            "fn(PIL.Image) -> HxW uint8 alpha. Refusing to return an opaque "
            "alpha, which would silently pass the whole frame through as "
            "foreground. See vendor-shims/rembg-shim."
        )

    if not isinstance(data, Image.Image):
        data = Image.fromarray(np.asarray(data))

    alpha = np.asarray(_provider(data))
    if alpha.ndim != 2:
        raise RembgShimError(
            f"mask provider returned shape {alpha.shape}; expected 2-D HxW alpha"
        )
    if alpha.shape != (data.height, data.width):
        raise RembgShimError(
            f"mask provider returned {alpha.shape}, expected "
            f"{(data.height, data.width)} to match the input image"
        )
    if alpha.dtype != np.uint8:
        alpha = (np.clip(alpha, 0, 1) * 255).astype(np.uint8) \
            if alpha.max() <= 1.0 else alpha.astype(np.uint8)

    rgba = data.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha, mode="L"))
    return rgba
