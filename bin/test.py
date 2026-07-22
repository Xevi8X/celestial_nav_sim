import matplotlib.pyplot as plt
from pathlib import Path

from celestial_nav import Navigator
from common import Io, Observer, Sky

path = Path(__file__).parent.parent / ".data" / "image.fit"

image, time, lat, lon = Io.load_fits(path)




plt.imshow(image, cmap="gray")
plt.show()

sky = Sky()
observer = Observer()
observer.set_time(time)
observer.set_location(latitude=lat, longitude=lon, elevation=0.0)

print("Image width: ", image.width, "px, height: ", image.height, "px")

navigator = Navigator(sky, fov_range=(8, 12), star_max_magnitude=7)
solution = navigator.find_solution(image)
print("Detected camera FOV: ", solution.fov)


