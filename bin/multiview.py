from celestial_nav import Navigator
from common import Observer, Sky
from common.ecef import ECEF
from sky_render import (
    Camera,
    CameraGeometry,
    CameraImageModel,
    ImageFormat,
    render,
)

import datetime
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__": 
    sky = Sky()
    camera = Camera(
        CameraGeometry(
            fov=35.0,
            width=2048,
            height=2048,
            image_format=ImageFormat.MONO8,
        ),
        CameraImageModel(fwhm=7.0),
    )
    renderer = render.Renderer(sky, camera)

    time = datetime.datetime(2026, 1, 1, 0, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=2)))

    observer = Observer()
    observer.set_time(time)
    observer.set_location(latitude=52.2297, longitude=21.0122, elevation=0.0)
    observer.set_look_direction(look_dir=[1.0, 0.0, -3], look_up=[0, 0, -1])

    data = []

    for _ in range(100):
        time += datetime.timedelta(seconds=1)
        observer.set_time(time)
        image = renderer.render(observer)
        data.append(Navigator.ImageTimeZenit(
            image=image.copy(),
            time=observer.time,
            zenit_cam=observer.observer_matrix @ np.array([0.0, 0.0, -1.0])
        ))

    navigator = Navigator(sky, fov_range=(30, 40), star_max_magnitude=7)
    location = navigator.estimate_location(data)
    location2 = navigator.estimate_location(data[0])


    print("Estimated location from zenit: ", location.latitude_deg, location.longitude_deg)
    print("Dist: ", np.linalg.norm(ECEF.north_east_vector(observer.latitude, observer.longitude, location.latitude_deg, location.longitude_deg)), "m")

    print("Estimated location2 from zenit: ", location2.latitude_deg, location2.longitude_deg)
    print("Dist: ", np.linalg.norm(ECEF.north_east_vector(observer.latitude, observer.longitude, location2.latitude_deg, location2.longitude_deg)), "m")
