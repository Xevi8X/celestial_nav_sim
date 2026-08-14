import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from common.config import GROUND_BACKGROUND_SCALE
from .camera import Camera
from .psf import (
    point_spread_kernel,
    point_spread_kernels,
    point_spread_support,
)


MAX_POINT_SOURCE_BATCH_SIZE = 512
POINT_SOURCE_BATCH_PIXELS = 500_000


class Canvas:
    def __init__(self, camera: Camera):
        if not camera.is_valid():
            raise ValueError("camera is not valid")
        self._camera = camera
        geometry = camera.geometry
        shape = (geometry.height, geometry.width)
        if not geometry.image_format.monochromatic:
            shape += (geometry.image_format.channels,)
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
        image_model = self._camera.image_model
        sigma = image_model.fwhm / 2.355
        is_point_source = 2 * radius_px <= image_model.fwhm
        blur_support = (
            point_spread_support(image_model.fwhm)
            if is_point_source
            else 4 * sigma
        )
        size = int(np.ceil(radius_px + blur_support + 1))
        cx, cy = int(np.floor(x)), int(np.floor(y))
        xs = np.arange(cx - size, cx + size + 1)
        ys = np.arange(cy - size, cy + size + 1)
        xx, yy = np.meshgrid(xs, ys)
        if is_point_source:
            kernel = point_spread_kernel(
                xx,
                yy,
                centroid_x=x,
                centroid_y=y,
                fwhm=image_model.fwhm,
            )
        else:
            distance = np.hypot(xx - x, yy - y)
            kernel = np.clip(radius_px + 0.5 - distance, 0, 1)
            kernel = gaussian_filter(kernel, sigma, mode="constant")
            kernel[distance > radius_px + 4 * sigma] = 0
            kernel /= kernel.sum()
        if exposure_time is None:
            exposure_time = image_model.exposure_time
        signal = exposure_time * image_model.flux * 10 ** (-0.4 * magnitude)
        h, w = self._image.shape[:2]
        x0, x1 = max(0, xs[0]), min(w, xs[-1] + 1)
        y0, y1 = max(0, ys[0]), min(h, ys[-1] + 1)
        if x0 >= x1 or y0 >= y1:
            return
        kernel = kernel[y0 - ys[0]:y1 - ys[0], x0 - xs[0]:x1 - xs[0]]
        if self._camera.geometry.image_format.monochromatic:
            self._image[y0:y1, x0:x1] += signal * kernel
        else:
            self._image[y0:y1, x0:x1] += (
                signal * kernel[:, :, None] * (np.asarray(color) / 255)
            )

    def draw_points(self, positions, magnitudes, exposure_time=None):
        """Draw point sources in batches while preserving source order."""
        positions = np.asarray(positions, dtype=np.float64)
        magnitudes = np.asarray(magnitudes, dtype=np.float64)
        if positions.size == 0 and magnitudes.size == 0:
            return
        if (
            positions.ndim != 2
            or positions.shape[1] != 2
            or magnitudes.ndim != 1
            or positions.shape[0] != magnitudes.size
        ):
            raise ValueError(
                "positions and magnitudes must have shapes (n, 2) and (n,)"
            )

        valid = np.isfinite(positions).all(axis=1) & np.isfinite(magnitudes)
        positions = positions[valid]
        magnitudes = magnitudes[valid]
        if positions.size == 0:
            return

        image_model = self._camera.image_model
        if exposure_time is None:
            exposure_time = image_model.exposure_time
        size = int(np.ceil(point_spread_support(image_model.fwhm) + 1))
        offsets = np.arange(-size, size + 1, dtype=np.float32)
        kernel_pixels = offsets.size ** 2
        batch_size = max(
            1,
            min(
                MAX_POINT_SOURCE_BATCH_SIZE,
                POINT_SOURCE_BATCH_PIXELS // kernel_pixels,
            ),
        )
        h, w = self._image.shape[:2]
        monochromatic = self._camera.geometry.image_format.monochromatic

        for start in range(0, len(positions), batch_size):
            stop = start + batch_size
            points = positions[start:stop]
            chunk_magnitudes = magnitudes[start:stop]
            centers = np.floor(points).astype(np.int64)
            fractions = (points - centers).astype(np.float32)
            coordinates = np.broadcast_to(
                offsets,
                (len(points), offsets.size),
            )
            kernels = point_spread_kernels(
                coordinates,
                coordinates,
                fractions[:, 0],
                fractions[:, 1],
                image_model.fwhm,
            )

            for (cx, cy), kernel, magnitude in zip(
                centers,
                kernels,
                chunk_magnitudes,
            ):
                xs0 = cx - size
                ys0 = cy - size
                x0, x1 = max(0, xs0), min(w, cx + size + 1)
                y0, y1 = max(0, ys0), min(h, cy + size + 1)
                if x0 >= x1 or y0 >= y1:
                    continue
                cropped_kernel = kernel[
                    y0 - ys0:y1 - ys0,
                    x0 - xs0:x1 - xs0,
                ]
                signal = (
                    exposure_time
                    * image_model.flux
                    * 10 ** (-0.4 * magnitude)
                )
                if monochromatic:
                    self._image[y0:y1, x0:x1] += signal * cropped_kernel
                else:
                    self._image[y0:y1, x0:x1] += (
                        signal * cropped_kernel[:, :, None]
                    )

    def add_horizon(self, observer_matrix):
        ground = self._camera.ground_mask(observer_matrix)
        image_model = self._camera.image_model
        sky_background = (
            image_model.exposure_time * image_model.sky_background
        )
        self._image += sky_background
        self._image[ground] = GROUND_BACKGROUND_SCALE * sky_background

    def add_noise(self, seed=None):
        rng = np.random.default_rng(seed)
        np.maximum(self._image, 0, out=self._image)
        self._image = rng.poisson(self._image).astype(np.float32)
        read_noise = self._camera.image_model.read_noise
        if read_noise > 0:
            self._image += rng.normal(
                0.0,
                read_noise,
                self._image.shape,
            ).astype(np.float32)

    def image(self):
        image_format = self._camera.geometry.image_format
        output = self._image * image_format.output_scale
        np.rint(output, out=output)
        np.clip(
            output,
            0,
            image_format.max_value,
            out=output,
        )
        output = output.astype(image_format.dtype)
        return Image.fromarray(output)

    def linear_image(self):
        return self._image.copy()

    def save(self, filename):
        self.image().save(filename)
