import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from .camera import Camera

class Canvas:
    def __init__(self, camera : Camera):
        self._camera = camera
        self._image = np.zeros((camera.height, camera.width, 3), dtype=np.float32)
        self._output = None
        self._horizon_added = False

    def draw(self, position, magnitude, radius_px=0.0, color=(255, 255, 255)):
        if self._output is not None:
            raise ValueError("Canvas already finalized")
        if self._horizon_added:
            raise ValueError("Cannot draw after adding the horizon")
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
        signal = self._camera.exposure_time * self._camera.flux * 10 ** (-0.4 * magnitude)
        h, w = self._image.shape[:2]
        x0, x1 = max(0, xs[0]), min(w, xs[-1] + 1)
        y0, y1 = max(0, ys[0]), min(h, ys[-1] + 1)
        if x0 >= x1 or y0 >= y1:
            return
        kernel = kernel[y0 - ys[0]:y1 - ys[0], x0 - xs[0]:x1 - xs[0]]
        self._image[y0:y1, x0:x1] += signal * kernel[:, :, None] * (np.asarray(color) / 255)

    def add_horizon(self, observer_matrix):
        if self._output is not None:
            raise ValueError("Canvas already finalized")
        if self._horizon_added:
            raise ValueError("Horizon already added")

        sky, ground = self._camera.horizon_fractions(observer_matrix)
        self._image *= sky[:, :, None]
        background_e = (
            self._camera.sky_background_e * sky
            + self._camera.ground_background_e * ground
        )
        self._image += background_e[:, :, None]
        self._horizon_added = True

    def add_noise(self, seed=None):
        if self._output is not None:
            raise ValueError("Noise already added")

        rng = np.random.default_rng(seed)

        if self._camera.monochromatic:
            expected_e = self._image[:, :, 0]
        else:
            expected_e = self._image

        measured_e = rng.poisson(np.maximum(expected_e, 0)).astype(np.float32)
        measured_e += rng.normal(
            0.0,
            self._camera.read_noise_e,
            measured_e.shape,
        ).astype(np.float32)
        self._output = np.rint(np.clip(measured_e, 0, 255)).astype(np.uint8)

    def image(self):
        if self._output is None:
            raise ValueError("Canvas not finalized. Call add_noise() first")
        return Image.fromarray(self._output)

    def save(self, filename):
        self.image().save(filename)
