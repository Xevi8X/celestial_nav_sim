from common import Observer, Sky
from sky_render import Camera, CameraGeometry, ImageFormat, Renderer
import datetime
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sky = Sky()
    camera = Camera(
        CameraGeometry(
            fov=80.0,
            width=800,
            height=640,
            image_format=ImageFormat.RGB8,
        ),
    )
    renderer = Renderer(sky, camera)

    observer = Observer()
    observer.set_location(latitude=52.2297, longitude=21.0122, elevation=0.0)

    observer.set_time(datetime.datetime(2026, 1, 1, 0, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=2))))
    observer.set_look_direction(look_dir=[1, 0.0, -0.8], look_up=[0, 0, -1])
    image = renderer.render(observer)

    plt.imshow(image)
    plt.title("Polar star")
    plt.axis("off")
    plt.show()

    observer.set_time(datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=2))))
    observer.set_look_direction(look_dir=[-1, 0.0, -0.5], look_up=[0, 0, -1])
    image = renderer.render(observer)

    plt.imshow(image)
    plt.title("Sun at noon")
    plt.axis("off")
    plt.show()
