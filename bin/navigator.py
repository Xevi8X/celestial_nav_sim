from celestial_nav import Navigator
from common import Observer, Sky
from common.ecef import ECEF
from sky_render import Camera, ImageFormat, render

import datetime
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__": 
    sky = Sky()
    camera = Camera(
        fov=35.0,
        width=2048,
        height=2048,
        image_format=ImageFormat.MONO8,
    )
    renderer = render.Renderer(sky, camera)

    observer = Observer()
    observer.set_time(datetime.datetime(2026, 1, 1, 0, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=2))))
    observer.set_location(latitude=52.2297, longitude=21.0122, elevation=0.0)
    observer.set_look_direction(look_dir=[0.0, -1.0, -3], look_up=[0, 0, -1])
    image = renderer.render(observer)

    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()


    navigator = Navigator(sky, fov_range=(30, 40), star_max_magnitude=7)
    solution = navigator.find_solution(image)
    print(solution)
    plt.imshow(solution.visual)
    plt.axis("off")
    plt.show()

    observer2 = navigator.estimate_orientation(image, time=observer.time, latitude_deg=observer.latitude, longitude_deg=observer.longitude, elevation_m=observer.elevation)

    print("Actual orientation: ", observer.look_dir, observer.look_up)
    print("Estimated orientation: ", observer2.look_dir, observer2.look_up)

    image_time_orientation = navigator.ImageTimeOrientation(
        image=image,
        time=observer.time,
        look_dir_ned=observer.look_dir,
        look_up_ned=observer.look_up
    )
    location = navigator.estimate_location_full_orientation(image_time_orientation)

    print("Actual location: ", observer.latitude, observer.longitude)
    print("Estimated location: ", location.latitude_deg, location.longitude_deg)
    print("Dist: ", np.linalg.norm(ECEF.north_east_vector(observer.latitude, observer.longitude, location.latitude_deg, location.longitude_deg)), "m")

    image_time_zenit = navigator.ImageTimeZenit(
        image=image,
        time=observer.time,
        zenit_cam=observer.observer_matrix @ np.array([0.0, 0.0, -1.0])
    )
    location = navigator.estimate_location(image_time_zenit)

    print("Estimated location from zenit: ", location.latitude_deg, location.longitude_deg)
    print("Dist: ", np.linalg.norm(ECEF.north_east_vector(observer.latitude, observer.longitude, location.latitude_deg, location.longitude_deg)), "m")
 
    
