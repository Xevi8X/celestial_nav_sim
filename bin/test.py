
import logging
from pathlib import Path
from common import Io, Sky

from sky_render import Calibrator


LOGGER = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    path = Path(__file__).parent.parent / ".data" / "image.fit"
    data = Io.load_fits(path)

    sky = Sky(magnitude_limit=10)
    calibrator = Calibrator(
        sky,
        exposure_time=0.215,
        visualize=True,
    )
    observer, camera = calibrator.estimate_camera(
        data.image,
        data.observation_time,
        data.latitude_deg,
        data.longitude_deg,
        elevation_m=data.elevation_m,
    )

    for report in calibrator.last_reports:
        LOGGER.info("%s", report)
    LOGGER.info("Observer: %s", observer)
    LOGGER.info("Camera: %s", camera)
