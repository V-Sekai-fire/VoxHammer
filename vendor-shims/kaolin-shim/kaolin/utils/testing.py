"""Shape/dtype/device validation, matching NVIDIA Kaolin's signature."""

from typing import Any, Optional, Sequence

import torch

__all__ = ["check_tensor"]


def check_tensor(
    tensor: Any,
    shape: Optional[Sequence[Optional[int]]] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Any] = None,
    throw: bool = True,
) -> bool:
    """Return whether ``tensor`` matches the given shape, dtype and device.

    A ``None`` entry in ``shape`` matches any size on that axis; the number of
    axes must still match.  With ``throw=True`` a mismatch raises; with
    ``throw=False`` it returns ``False``, which is how FlexiCubes calls it.
    """
    def _fail(msg: str) -> bool:
        if throw:
            raise ValueError(msg)
        return False

    if not torch.is_tensor(tensor):
        return _fail(f"expected a torch.Tensor, got {type(tensor).__name__}")

    if shape is not None:
        if tensor.dim() != len(shape):
            return _fail(
                f"expected {len(shape)} dimensions, got {tensor.dim()} "
                f"(shape {tuple(tensor.shape)})"
            )
        for axis, (want, have) in enumerate(zip(shape, tensor.shape)):
            if want is not None and want != have:
                return _fail(
                    f"expected size {want} on axis {axis}, got {have} "
                    f"(shape {tuple(tensor.shape)})"
                )

    if dtype is not None and tensor.dtype != dtype:
        return _fail(f"expected dtype {dtype}, got {tensor.dtype}")

    if device is not None and torch.device(device).type != tensor.device.type:
        return _fail(f"expected device {device}, got {tensor.device}")

    return True
