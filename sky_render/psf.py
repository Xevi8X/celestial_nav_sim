import math

import numpy as np


MOFFAT_BETA = 1.5
MOFFAT_TAPER_ALPHA = 6.0
MOFFAT_SUPPORT_ALPHA = 8.0


def moffat_alpha(fwhm: float) -> float:
    """Convert Moffat FWHM to its radial alpha scale."""
    return fwhm / (
        2 * math.sqrt(2 ** (1 / MOFFAT_BETA) - 1)
    )


def point_spread_support(fwhm: float) -> float:
    """Finite support used consistently by fitting and rendering."""
    return MOFFAT_SUPPORT_ALPHA * moffat_alpha(fwhm)


def point_spread_taper_start(fwhm: float) -> float:
    """Radius where the smooth transition to zero begins."""
    return MOFFAT_TAPER_ALPHA * moffat_alpha(fwhm)


def point_spread_kernel(
    xx: np.ndarray,
    yy: np.ndarray,
    centroid_x: float,
    centroid_y: float,
    fwhm: float,
) -> np.ndarray:
    """Return a normalized, finite-support Moffat point-spread function."""
    alpha = moffat_alpha(fwhm)
    radius = np.hypot(xx - centroid_x, yy - centroid_y)
    kernel = (1 + (radius / alpha) ** 2) ** (-MOFFAT_BETA)
    taper_start = point_spread_taper_start(fwhm)
    support = point_spread_support(fwhm)
    taper_phase = np.clip(
        (radius - taper_start) / (support - taper_start),
        0,
        1,
    )
    taper = 0.5 * (1 + np.cos(np.pi * taper_phase))
    kernel *= taper
    kernel[radius >= support] = 0
    total = float(np.sum(kernel))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("FWHM produced an empty point-spread function")
    return kernel / total


def source_mask_radius(fwhm: float) -> int:
    """Radius sufficient to exclude visible Moffat wings from sky samples."""
    return max(2, int(math.ceil(4 * fwhm)))
