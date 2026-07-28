
from pathlib import Path
from common import Io, Sky

from tools import Calibrator


if __name__ == "__main__":
    path = Path(__file__).parent.parent / ".data" / "image.fit"
    image, time, lat, lon = Io.load_fits(path)

    sky = Sky(magnitude_limit=10)
    calibrator = Calibrator(sky)
    observer, camera = calibrator.estimate_camera(image, time, lat, lon, elevation_m=0.0, exposure_time=0.215)

    print(observer)
    print(camera)

