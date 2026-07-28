import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from common.config import GROUND_BACKGROUND_SCALE
from .camera import Camera


class Canvas:
    def __init__(self, camera: Camera):
        self._camera = camera
        shape = (camera.height, camera.width)
        if not camera.image_format.monochromatic:
            shape += (camera.image_format.channels,)
        self._image = np.zeros(shape, dtype=np.float32)

    def draw(
        self,
        position,
        magnitude,
        radius_px=0.0,
        color=(255, 255, 255),
        exposure_time=None,
    ):
        x, y = position
        if not np.isfinite([x, y, magnitude, radius_px]).all() or radius_px < 0:
            return
        sigma = self._camera.fwhm / 2.355
        size = int(np.ceil(radius_px + 4 * sigma + 1))
        cx, cy = int(np.floor(x)), int(np.floor(y))
        xs = np.arange(cx - size, cx + size + 1)
        ys = np.arange(cy - size, cy + size + 1)
        xx, yy = np.meshgrid(xs, ys)
        if 2 * radius_px <= self._camera.fwhm:
            kernel = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        else:
            kernel = np.clip(radius_px + 0.5 - np.hypot(xx - x, yy - y), 0, 1)
            kernel = gaussian_filter(kernel, sigma, mode="constant")
        kernel[np.hypot(xx - x, yy - y) > radius_px + 4 * sigma] = 0
        kernel /= kernel.sum()
        if exposure_time is None:
            exposure_time = self._camera.exposure_time
        signal = exposure_time * self._camera.flux * 10 ** (-0.4 * magnitude)
        h, w = self._image.shape[:2]
        x0, x1 = max(0, xs[0]), min(w, xs[-1] + 1)
        y0, y1 = max(0, ys[0]), min(h, ys[-1] + 1)
        if x0 >= x1 or y0 >= y1:
            return
        kernel = kernel[y0 - ys[0]:y1 - ys[0], x0 - xs[0]:x1 - xs[0]]
        if self._camera.image_format.monochromatic:
            self._image[y0:y1, x0:x1] += signal * kernel
        else:
            self._image[y0:y1, x0:x1] += (
                signal * kernel[:, :, None] * (np.asarray(color) / 255)
            )

    def add_horizon(self, observer_matrix):
        ground = self._camera.ground_mask(observer_matrix)
        sky_background = self._camera.exposure_time * self._camera.sky_background
        self._image[~ground] += sky_background
        self._image[ground] = GROUND_BACKGROUND_SCALE * sky_background

    def add_noise(self, seed=None):
        rng = np.random.default_rng(seed)
        self._image = rng.poisson(np.maximum(self._image, 0)).astype(np.float32)
        self._image += rng.normal(
            0.0,
            self._camera.read_noise,
            self._image.shape,
        ).astype(np.float32)

    def image(self):
        image_format = self._camera.image_format
        output = self._image * image_format.output_scale
        output = np.clip(output, 0, image_format.max_value).astype(image_format.dtype)
        return Image.fromarray(output)

    def linear_image(self):
        return self._image.copy()

    def save(self, filename):
        self.image().save(filename)
