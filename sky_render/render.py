import numpy as np

from .camera import Camera
from .canvas import Canvas
from common import Sky
from common import Observer

RENDER_COLORS = {
    "Sun": (255, 255, 0),
    "Moon": (173, 216, 230),
    "Mercury": (169, 169, 169),
    "Venus": (255, 255, 224),
    "Mars": (255, 0, 0),
    "Jupiter": (255, 165, 0),
    "Saturn": (210, 180, 140),
    "Uranus": (0, 255, 255),
    "Neptune": (0, 0, 255),
    "Pluto": (255, 192, 203),
}

def render(sky: Sky, observer: Observer, camera: Camera):
    stars_info, stars_ned = sky.get_stars_ned(observer)
    bodies_info, bodies_ned = sky.get_bodies_ned(observer)
    matrix = observer.observer_matrix.T @ camera.camera_matrix.T

    def _project(vectors):
        projected = vectors @ matrix
        with np.errstate(divide="ignore", invalid="ignore"):
            return projected[:, :2] / projected[:, 2, None], projected[:, 2]

    star_xy, star_z = _project(stars_ned)
    body_xy, body_z = _project(bodies_ned)

    def visible(ned, xy, z):
        return (
            (ned[:, 2] < 0) &
            (z > 0) &
            np.isfinite(xy).all(axis=1) &
            (xy[:, 0] >= 0) & (xy[:, 0] < camera.width) &
            (xy[:, 1] >= 0) & (xy[:, 1] < camera.height)
        )

    stars_visible = visible(stars_ned, star_xy, star_z)
    bodies_visible = visible(bodies_ned, body_xy, body_z)
    canvas = Canvas(camera)

    for point, info in zip(
        star_xy[stars_visible],
        np.asarray(stars_info, dtype=object)[stars_visible]
    ):
        canvas.draw(point, info.magnitude)


    for point, vector, info in zip(
        body_xy[bodies_visible],
        bodies_ned[bodies_visible],
        np.asarray(bodies_info, dtype=object)[bodies_visible]
    ):
        distance = np.linalg.norm(vector)
        angular_radius = np.arcsin(np.clip(info.radius_km / distance, 0, 1))
        radius_px = camera.focal_length * np.tan(angular_radius)
        color = RENDER_COLORS.get(info.name, (255, 255, 255)) if camera.monochromatic else (255, 255, 255)
        canvas.draw(point, info.apparent_magnitude, radius_px, color)

    return canvas
