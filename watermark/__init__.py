"""
Watermark package – baseline (Wen et al. 2023) and three novel methods.
"""

from watermark.baseline import TreeRingWatermark
from watermark.novel import MultiBitWatermark, LogPolarWatermark, EnsembleWatermark

__all__ = [
    "TreeRingWatermark",
    "MultiBitWatermark",
    "LogPolarWatermark",
    "EnsembleWatermark",
]
