from dataclasses import fields
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from sky_render import Camera, ImageFormat, Renderer
from sky_render.canvas import Canvas


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
        fov=60.0,
        width=WIDTH,
        height=HEIGHT,
        exposure_time=0.5,
        flux=100.0,
        fwhm=1.0,
        sky_background=12.0,
        read_noise=1.0,
        image_format=image_format,
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
            ImageFormat.MONO8: (1, np.uint8, 255, 1.0, True),
            ImageFormat.MONO16: (1, np.uint16, 65535, 257.0, True),
            ImageFormat.RGB8: (3, np.uint8, 255, 1.0, False),
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
                        image_format.monochromatic,
                    ),
                    properties,
                )

        self.assertIs(
            Camera(fov=60.0, width=WIDTH, height=HEIGHT).image_format,
            ImageFormat.MONO8,
        )
        self.assertEqual(
            [field.name for field in fields(Camera)],
            [
                "fov",
                "width",
                "height",
                "exposure_time",
                "flux",
                "fwhm",
                "sky_background",
                "read_noise",
                "image_format",
            ],
        )


class RenderFormatTests(unittest.TestCase):
    def setUp(self):
        self.observer = SimpleNamespace(
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

    def test_horizon_mask_is_sharp_and_ground_background_is_scaled(self):
        camera = Camera(
            fov=90.0,
            width=5,
            height=5,
            exposure_time=2.0,
            sky_background=4.0,
            read_noise=0.0,
            image_format=ImageFormat.MONO8,
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
