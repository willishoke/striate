# Enable CPU fallback for MPS ops that aren't implemented on Apple Silicon
# (FFT and grid_sample, the two ops at the heart of feature viz, may fall back).
# Must be set before torch is imported anywhere in the package.
import os as _os

_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__all__ = ["model", "objectives", "param", "render", "transforms", "util"]
