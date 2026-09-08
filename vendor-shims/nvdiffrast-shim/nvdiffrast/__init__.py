"""NOT NVlabs nvdiffrast. Import-only stand-in that raises on use.

Why: trellis/utils/postprocessing_utils.py imports nvdiffrast.torch at module
level, and voxhammer/edit_pipeline.py imports that module, so nvdiffrast must
be importable merely to reach the edit path. It is actually *used* only in
bake_texture (dr.texture, uv_dr), which to_glb calls.

Real nvdiffrast compiles a CUDA extension whose toolkit version must match
torch's. This environment pins torch 2.4.0+cu118 while the system toolkit is
12.4, and no CUDA 11.8 nvcc is available from conda-forge for win-64, so the
real package cannot build here.

This shim deliberately RAISES on every call rather than returning a no-op.
A silent no-op would let a texture-baking failure masquerade as a successful
run and produce an untextured mesh that looks like a result. If you see
NvdiffrastShimError, the geometry path genuinely needs texture baking and this
environment cannot provide it -- use the container route (RFD 0036,
weftspun/trellis2-base) or move the stack to a CUDA line with a matching nvcc.
"""

__is_shim__ = True


class NvdiffrastShimError(RuntimeError):
    """Raised when shimmed nvdiffrast is actually called."""


def _refuse(name):
    def _fn(*args, **kwargs):
        raise NvdiffrastShimError(
            f"nvdiffrast.{name}() was called, but this is the import-only shim. "
            "Texture baking is genuinely required here; the shim exists only so "
            "the edit path can be imported. See vendor-shims/nvdiffrast-shim."
        )
    return _fn
