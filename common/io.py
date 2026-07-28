from pathlib import Path
from typing import Union
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from astropy.io import fits

class Io:
    @staticmethod
    def load_fits(path: Union[str, Path]) -> tuple[Image.Image, datetime, float, float]:
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

        time = datetime.fromisoformat(header["DATE-OBS"].replace("Z", "+00:00"))
        if time.tzinfo is None:
            time = time.replace(tzinfo=timezone.utc)

        return image, time, float(header["LAT-OBS"]), float(header["LONG-OBS"])
