"""Refusing stand-ins for the nvdiffrast.torch surface TRELLIS touches."""

from .. import NvdiffrastShimError, _refuse

__all__ = [
    "RasterizeCudaContext", "RasterizeGLContext",
    "rasterize", "interpolate", "texture", "antialias",
    "NvdiffrastShimError",
]

rasterize = _refuse("torch.rasterize")
interpolate = _refuse("torch.interpolate")
texture = _refuse("torch.texture")
antialias = _refuse("torch.antialias")


class _RefusingContext:
    def __init__(self, *args, **kwargs):
        raise NvdiffrastShimError(
            "nvdiffrast rasterizer context requested, but this is the "
            "import-only shim. See vendor-shims/nvdiffrast-shim."
        )


RasterizeCudaContext = _RefusingContext
RasterizeGLContext = _RefusingContext
