

from common import Observer, Sky
from sky_render import Camera, render
import datetime
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sky = Sky()
    observer = Observer()
    observer.set_time(datetime.datetime(2026, 1, 1, 0, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=2))))
    observer.set_location(latitude=52.2297, longitude=21.0122, elevation=0.0)
    observer.set_look_direction(look_dir=[1, 0.45, -0.9], look_up=[0, 0, -1])
    camera = Camera(fov=60.0, width=800, height=640, monochromatic=False)
    canvas = render(sky, observer, camera)
    canvas.save("Polaris.png")
    print(canvas.image())
    plt.imshow(canvas.image(), cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.show()