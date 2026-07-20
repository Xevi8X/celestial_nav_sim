from celestial_nav import Navigator
from common import Observer, Sky
from common.ecef import ECEF
from sky_render import Camera, render

import datetime
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__": 
    sky = Sky()
    camera = Camera(fov=35.0, width=800, height=800, monochromatic=True)
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

    observer_no_orientation = Observer()
    observer_no_orientation.set_time(observer.time)
    observer_no_orientation.set_location(latitude=observer.latitude, longitude=observer.longitude, elevation=observer.elevation)
    observer2 = navigator.estimate_orientation(observer_no_orientation, image)

    print("Actual orientation: ", observer.look_dir, observer.look_up)
    print("Estimated orientation: ", observer2.look_dir, observer2.look_up)

    observer_no_location = Observer()
    observer_no_location.set_time(observer.time)
    observer_no_location.set_look_direction(look_dir=observer.look_dir, look_up=observer.look_up)
    observer3 = navigator.estimate_location(observer_no_location, image)

    print("Actual location: ", observer.latitude, observer.longitude)
    print("Estimated location: ", observer3.latitude, observer3.longitude)
    print("Dist: ", np.linalg.norm(ECEF.north_east_vector(observer.latitude, observer.longitude, observer3.latitude, observer3.longitude)), "m")
 
    


