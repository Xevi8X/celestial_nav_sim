import datetime
from dataclasses import fields
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from sky_render import (
    Camera,
    CameraGeometry,
    CameraImageModel,
    ImageFormat,
    Renderer,
)
from sky_render.canvas import Canvas
from sky_render.psf import (
    point_spread_kernel,
    point_spread_support,
    point_spread_taper_start,
)


WIDTH = 9
HEIGHT = 7
SKY_MATRIX = np.diag([1.0, -1.0, -1.0])
HORIZONTAL_MATRIX = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
])


class EmptySky:
    def get_stars_ecef(self, observer):
        return [], np.empty((0, 3), dtype=np.float64)

    def get_bodies_ecef(self, observer):
        return [], np.empty((0, 3), dtype=np.float64)


class NoiseFreeRng:
    def poisson(self, expected):
        return np.asarray(expected)

    def normal(self, mean, standard_deviation, size):
        return np.zeros(size, dtype=np.float32)


def make_camera(image_format):
    return Camera(
        CameraGeometry(
            fov=60.0,
            width=WIDTH,
            height=HEIGHT,
            image_format=image_format,
        ),
        CameraImageModel(
            exposure_time=0.5,
            flux=100.0,
            fwhm=1.0,
            sky_background=12.0,
            read_noise=1.0,
        ),
    )


def make_finalized_canvas(image_format, seed=123):
    canvas = Canvas(make_camera(image_format))
    canvas.draw((4.2, 3.1), magnitude=2.0)
    canvas.add_horizon(SKY_MATRIX)
    canvas.add_noise(seed)
    return canvas


class ImageFormatTests(unittest.TestCase):
    def test_properties(self):
        expected = {
            ImageFormat.MONO8: (1, np.uint8, 255, 1.0, 1.0, True),
            ImageFormat.MONO16: (1, np.uint16, 65535, 1.0, 257.0, True),
            ImageFormat.RGB8: (3, np.uint8, 255, 1.0, 1.0, False),
        }

        self.assertEqual(set(ImageFormat), set(expected))
        for image_format, properties in expected.items():
            with self.subTest(image_format=image_format):
                self.assertEqual(
                    (
                        image_format.channels,
                        image_format.dtype,
                        image_format.max_value,
                        image_format.output_scale,
                        image_format.navigation_scale,
                        image_format.monochromatic,
                    ),
                    properties,
                )

        self.assertIs(
            CameraGeometry(fov=60.0, width=WIDTH, height=HEIGHT).image_format,
            ImageFormat.MONO8,
        )
        self.assertEqual(
            [field.name for field in fields(Camera)],
            ["geometry", "image_model"],
        )
        self.assertEqual(
            [field.name for field in fields(CameraGeometry)],
            [
                "fov",
                "width",
                "height",
                "image_format",
            ],
        )
        self.assertEqual(
            [field.name for field in fields(CameraImageModel)],
            [
                "exposure_time",
                "flux",
                "fwhm",
                "sky_background",
                "read_noise",
            ],
        )


class RenderFormatTests(unittest.TestCase):
    def setUp(self):
        self.observer = SimpleNamespace(
            time=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
            latitude=0.0,
            longitude=0.0,
            observer_matrix=SKY_MATRIX,
        )

    def test_renderer_output_contract_and_same_seed(self):
        expected = {
            ImageFormat.MONO8: ("L", np.uint8, (HEIGHT, WIDTH)),
            ImageFormat.MONO16: ("I;16", np.uint16, (HEIGHT, WIDTH)),
            ImageFormat.RGB8: ("RGB", np.uint8, (HEIGHT, WIDTH, 3)),
        }

        for image_format, (mode, dtype, shape) in expected.items():
            with self.subTest(image_format=image_format):
                renderer = Renderer(EmptySky(), make_camera(image_format))
                first = renderer.render(self.observer, noise_seed=123)
                second = renderer.render(self.observer, noise_seed=123)
                first_array = np.asarray(first)
                second_array = np.asarray(second)

                self.assertIsInstance(first, Image.Image)
                self.assertEqual(first.mode, mode)
                self.assertEqual(first.size, (WIDTH, HEIGHT))
                self.assertEqual(first_array.dtype, np.dtype(dtype))
                self.assertEqual(first_array.shape, shape)
                np.testing.assert_array_equal(first_array, second_array)

                if image_format is ImageFormat.RGB8:
                    self.assertFalse(
                        np.array_equal(first_array[:, :, 0], first_array[:, :, 1])
                    )

    def test_image_and_save_do_not_regenerate_noise(self):
        for image_format in ImageFormat:
            with self.subTest(image_format=image_format):
                canvas = make_finalized_canvas(image_format)
                expected = np.array(canvas.image())

                np.testing.assert_array_equal(expected, np.asarray(canvas.image()))
                np.testing.assert_array_equal(expected, np.asarray(canvas.image()))

                with tempfile.TemporaryDirectory() as directory:
                    first_path = Path(directory) / "first.png"
                    second_path = Path(directory) / "second.png"
                    canvas.save(first_path)
                    canvas.save(second_path)

                    with Image.open(first_path) as first_saved:
                        first_array = np.array(first_saved, dtype=expected.dtype)
                    with Image.open(second_path) as second_saved:
                        second_array = np.array(second_saved, dtype=expected.dtype)

                np.testing.assert_array_equal(expected, first_array)
                np.testing.assert_array_equal(expected, second_array)
                np.testing.assert_array_equal(expected, np.asarray(canvas.image()))

    def test_image_does_not_require_horizon_or_noise(self):
        expected = {
            ImageFormat.MONO8: ("L", np.uint8, (HEIGHT, WIDTH)),
            ImageFormat.MONO16: ("I;16", np.uint16, (HEIGHT, WIDTH)),
            ImageFormat.RGB8: ("RGB", np.uint8, (HEIGHT, WIDTH, 3)),
        }

        for image_format, (mode, dtype, shape) in expected.items():
            with self.subTest(image_format=image_format):
                canvas = Canvas(make_camera(image_format))
                image = canvas.image()
                array = np.asarray(image)

                self.assertEqual(image.mode, mode)
                self.assertEqual(image.size, (WIDTH, HEIGHT))
                self.assertEqual(array.dtype, np.dtype(dtype))
                self.assertEqual(array.shape, shape)
                self.assertFalse(array.any())

    def test_mono16_noise_uses_native_count_units(self):
        expected_background = 400.0
        expected_read_noise = 5.0
        camera = Camera(
            CameraGeometry(
                fov=60.0,
                width=256,
                height=256,
                image_format=ImageFormat.MONO16,
            ),
            CameraImageModel(
                exposure_time=1.0,
                sky_background=expected_background,
                read_noise=expected_read_noise,
            ),
        )
        self.assertFalse(camera.ground_mask(SKY_MATRIX).any())

        canvas = Canvas(camera)
        canvas.add_horizon(SKY_MATRIX)
        canvas.add_noise(seed=123)
        values = np.asarray(canvas.image(), dtype=np.float64)

        self.assertAlmostEqual(
            float(np.median(values)),
            expected_background,
            delta=2.0,
        )
        self.assertAlmostEqual(
            float(np.std(values)),
            np.sqrt(expected_background + expected_read_noise ** 2),
            delta=1.0,
        )
        self.assertGreater(np.unique(values).size, 100)
        self.assertTrue(np.any(values % 257))

    def test_point_sources_have_normalized_moffat_wings(self):
        yy, xx = np.indices((61, 61), dtype=np.float64)
        fwhm = 4.0
        kernel = point_spread_kernel(
            xx,
            yy,
            centroid_x=30.25,
            centroid_y=29.75,
            fwhm=fwhm,
        )

        self.assertAlmostEqual(float(np.sum(kernel)), 1.0)
        self.assertGreater(kernel[30, 42], 0.0)
        taper_start = int(np.ceil(point_spread_taper_start(fwhm)))
        edge = int(np.floor(point_spread_support(fwhm)))
        self.assertLess(
            kernel[30, 30 + edge],
            kernel[30, 30 + taper_start] * 0.1,
        )
        outside = int(np.ceil(point_spread_support(fwhm))) + 1
        self.assertEqual(kernel[30, 30 + outside], 0.0)

    def test_horizon_mask_is_sharp_and_ground_background_is_scaled(self):
        camera = Camera(
            CameraGeometry(
                fov=90.0,
                width=5,
                height=5,
                image_format=ImageFormat.MONO8,
            ),
            CameraImageModel(
                exposure_time=2.0,
                sky_background=4.0,
                read_noise=0.0,
            ),
        )
        ground = camera.ground_mask(HORIZONTAL_MATRIX)
        expected_ground = np.array(
            [
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ]
        )
        np.testing.assert_array_equal(ground, expected_ground)

        canvas = Canvas(camera)
        canvas.add_horizon(HORIZONTAL_MATRIX)
        with patch(
            "sky_render.canvas.np.random.default_rng",
            return_value=NoiseFreeRng(),
        ):
            canvas.add_noise(seed=123)

        expected_background = np.where(ground, 16, 8).astype(np.uint8)
        np.testing.assert_array_equal(np.asarray(canvas.image()), expected_background)


if __name__ == "__main__":
    unittest.main()
