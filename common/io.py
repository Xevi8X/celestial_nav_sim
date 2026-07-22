from pathlib import Path
from typing import Union
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from astropy.io import fits
from astropy.visualization import ImageNormalize, ZScaleInterval, SqrtStretch

class Io:
    @staticmethod
    def load_fits(path: Union[str, Path]) -> tuple[Image.Image, datetime, float, float]:
        with fits.open(path, lazy_load_hdus=True) as hdul:
            data = hdul[0].data.astype(np.uint16)
            header = hdul[0].header

        image = Image.fromarray(
            np.flipud((data // 257).astype(np.uint8))
        )

        time = datetime.fromisoformat(header["DATE-OBS"].replace("Z", "+00:00"))
        if time.tzinfo is None:
            time = time.replace(tzinfo=timezone.utc)

        return image, time, float(header["LAT-OBS"]), float(header["LONG-OBS"])