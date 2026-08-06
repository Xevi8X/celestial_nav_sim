from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image
from astropy.io import fits


@dataclass
class ImageData:
    """Image pixels and the acquisition metadata used by navigation."""

    image: Image.Image
    observation_time: datetime
    exposure_s: float
    latitude_deg: Optional[float]
    longitude_deg: Optional[float]
    elevation_m: float


class Io:
    @staticmethod
    def load_fits(path: Union[str, Path]) -> ImageData:
        with fits.open(path, lazy_load_hdus=True) as hdul:
            data = np.asarray(hdul[0].data)
            header = hdul[0].header

        if not np.isfinite(data).all():
            raise ValueError("FITS image contains non-finite values")
        if np.issubdtype(data.dtype, np.integer) and data.dtype.itemsize == 1:
            data = np.clip(data, 0, 255).astype(np.uint8)
        else:
            data = np.clip(data, 0, 65535).astype(np.uint16)

        image = Image.fromarray(data)

        if "DATE-OBS" not in header:
            raise ValueError("FITS header does not contain DATE-OBS")
        time = datetime.fromisoformat(
            str(header["DATE-OBS"]).replace("Z", "+00:00")
        )
        if time.tzinfo is None:
            time = time.replace(tzinfo=timezone.utc)

        lat = float(header["LAT-OBS"]) if "LAT-OBS" in header else None
        lon = float(header["LONG-OBS"]) if "LONG-OBS" in header else None
        exposure = header.get("EXPTIME", header.get("EXPOSURE"))
        if exposure is None:
            raise ValueError("FITS header does not contain EXPTIME or EXPOSURE")
        exposure = float(exposure)
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError("FITS exposure must be positive and finite")
        elevation = float(header.get(
            "ALT-OBS",
            header.get("ELEVATIO", header.get("ELEVATION", 0.0)),
        ))
        if not np.isfinite(elevation):
            raise ValueError("FITS observer elevation must be finite")

        return ImageData(
            image=image,
            observation_time=time,
            exposure_s=exposure,
            latitude_deg=lat,
            longitude_deg=lon,
            elevation_m=elevation,
        )
