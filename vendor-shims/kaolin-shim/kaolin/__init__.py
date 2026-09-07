"""NOT NVIDIA Kaolin.

This is a shim providing only ``kaolin.utils.testing.check_tensor``, the single
symbol TRELLIS's FlexiCubes imports. If you need real Kaolin -- meshes, camera
models, rendering, metrics, datasets -- this package does not provide it and
will not error helpfully when you reach for it. Uninstall this shim and install
NVIDIA Kaolin instead.

Why it exists: FlexiCubes uses ``check_tensor`` purely as a shape assertion
(always with ``throw=False``, inside ``assert`` statements). Real Kaolin
publishes wheels per torch/CUDA combination and none exists for
torch 2.4.0 + cu118, which is what VoxHammer pins.
"""

__all__ = ["utils"]
__is_shim__ = True
