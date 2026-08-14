from celestial_nav import Navigator
from common import Observer, Sky
from common.ecef import ECEF
from sky_render import Camera, CameraGeometry, ImageFormat, render

import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np


LOGGER = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    sky = Sky()
    camera = Camera(
        CameraGeometry(
            fov=35.0,
            width=1024,
            height=1024,
            image_format=ImageFormat.MONO8,
        ),
    )
    renderer = render.Renderer(sky, camera)

    observer = Observer()
    observer.set_time(datetime.datetime(2026, 1, 1, 0, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=2))))
    observer.set_location(latitude=52.2297, longitude=21.0122, elevation=0.0)
    observer.set_look_direction(look_dir=[-1.0, 0.0, -5], look_up=[0, 0, -1])
    image = renderer.render(observer, noise_seed = 42)

    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()


    navigator = Navigator(sky, fov_range=(30, 40), star_max_magnitude=7)
    solution = navigator.find_solution(image)
    LOGGER.info("Plate solution: %s", solution)
    plt.imshow(solution.visual)
    plt.axis("off")
    plt.show()

    observer2 = navigator.estimate_orientation(image, time=observer.time, latitude_deg=observer.latitude, longitude_deg=observer.longitude, elevation_m=observer.elevation)

    LOGGER.info("Actual orientation: %s %s", observer.look_dir, observer.look_up)
    LOGGER.info(
        "Estimated orientation: %s %s",
        observer2.look_dir,
        observer2.look_up,
    )

    image_time_orientation = navigator.ImageTimeOrientation(
        image=image,
        time=observer.time,
        look_dir_ned=observer.look_dir,
        look_up_ned=observer.look_up
    )
    location = navigator.estimate_location_full_orientation(image_time_orientation)

    distance = np.linalg.norm(ECEF.north_east_vector(
        observer.latitude,
        observer.longitude,
        location.latitude_deg,
        location.longitude_deg,
    ))
    LOGGER.info("Actual location: %.6f, %.6f", observer.latitude, observer.longitude)
    LOGGER.info(
        "Estimated location: %.6f, %.6f (error %.1f m)",
        location.latitude_deg,
        location.longitude_deg,
        distance,
    )

    image_time_zenit = navigator.ImageTimeZenit(
        image=image,
        time=observer.time,
        zenit_cam=observer.observer_matrix @ np.array([0.0, 0.0, -1.0])
    )
    location = navigator.estimate_location(image_time_zenit)

    distance = np.linalg.norm(ECEF.north_east_vector(
        observer.latitude,
        observer.longitude,
        location.latitude_deg,
        location.longitude_deg,
    ))
    LOGGER.info(
        "Estimated zenith location: %.6f, %.6f (error %.1f m)",
        location.latitude_deg,
        location.longitude_deg,
        distance,
    )
 
    
