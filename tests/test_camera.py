import unittest

import numpy as np

from sky_render import (
    Camera,
    CameraGeometry,
    CameraImageModel,
    ImageFormat,
)


class CameraLifecycleTests(unittest.TestCase):
    def test_empty_camera_can_be_filled_incrementally(self):
        camera = Camera()

        self.assertFalse(camera.is_valid())

        camera.set_image(640, 480, ImageFormat.MONO16)
        self.assertFalse(camera.is_valid())

        camera.set_fov(60.0)
        self.assertTrue(camera.is_valid())
        self.assertEqual(
            (
                camera.geometry.width,
                camera.geometry.height,
                camera.geometry.image_format,
            ),
            (640, 480, ImageFormat.MONO16),
        )

    def test_camera_has_separate_geometry_and_image_model(self):
        geometry = CameraGeometry(
            fov=60.0,
            width=640,
            height=480,
            image_format=ImageFormat.RGB8,
        )
        image_model = CameraImageModel(
            exposure_time=0.25,
            flux=1234.0,
            fwhm=2.5,
            sky_background=8.0,
            read_noise=0.75,
        )
        camera = Camera(geometry, image_model)

        self.assertTrue(camera.is_valid())
        self.assertIs(camera.geometry, geometry)
        self.assertIs(camera.image_model, image_model)

    def test_setters_assign_image_model_values(self):
        camera = Camera()

        camera.set_exposure_time(0.25)
        camera.set_photometry(flux=1234.0, fwhm=2.5)
        camera.set_noise(sky_background=8.0, read_noise=0.75)

        self.assertEqual(
            camera.image_model,
            CameraImageModel(
                exposure_time=0.25,
                flux=1234.0,
                fwhm=2.5,
                sky_background=8.0,
                read_noise=0.75,
            ),
        )

    def test_setters_do_not_validate_each_assignment(self):
        camera = Camera(
            CameraGeometry(fov=60.0, width=640, height=480),
        )

        camera.set_fov(180.0)
        self.assertEqual(camera.geometry.fov, 180.0)
        self.assertFalse(camera.is_valid())

        camera.set_photometry(flux=100.0, fwhm=0.0)
        self.assertEqual(camera.image_model.fwhm, 0.0)
        self.assertFalse(camera.is_valid())

    def test_is_valid_checks_both_groups(self):
        camera = Camera(
            CameraGeometry(fov=60.0, width=640, height=480),
        )
        self.assertTrue(camera.is_valid())

        camera.geometry.image_format = "L"
        self.assertFalse(camera.is_valid())

        camera.geometry.image_format = ImageFormat.MONO8
        camera.image_model.read_noise = -1.0
        self.assertFalse(camera.is_valid())

    def test_derived_geometry_rejects_invalid_camera(self):
        camera = Camera()

        with self.assertRaisesRegex(ValueError, "camera is not valid"):
            _ = camera.focal_length
        with self.assertRaisesRegex(ValueError, "camera is not valid"):
            _ = camera.camera_matrix
        with self.assertRaisesRegex(ValueError, "camera is not valid"):
            camera.ground_mask(np.eye(3))

    def test_derived_geometry(self):
        camera = Camera(
            CameraGeometry(fov=90.0, width=5, height=3),
        )

        self.assertAlmostEqual(camera.focal_length, 2.5)
        np.testing.assert_allclose(
            camera.camera_matrix,
            np.array([
                [2.5, 0.0, 2.0],
                [0.0, -2.5, 1.0],
                [0.0, 0.0, 1.0],
            ]),
        )


if __name__ == "__main__":
    unittest.main()
