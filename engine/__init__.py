from .core import train_one_epoch
from .core import validate
from .postprocess import heatmap_to_coords


# 点名HOOK

from . import hooks
from .builder import build_hooks
__all__ = ['build_hooks']
