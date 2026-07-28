import datetime
import unittest

import numpy as np
from PIL import Image

from celestial_nav import LostInSpace
from common import Observer, Sky
from sky_render import (
    Camera,
    CameraGeometry,
    CameraImageModel,
    ImageFormat,
)
from tools import (
    CalibrationConfig,
    CalibrationContext,
    CalibrationError,
    CalibrationStep,
    CalibrationVisualizer,
    Calibrator,
    StarMeasurement,
    otsu_threshold,
)


UTC_TIME = datetime.datetime(
    2026,
    7,
    28,
    tzinfo=datetime.timezone.utc,
)


def empty_sky():
    return Sky.__new__(Sky)


def configure_pair(events, name):
    def step(observer, camera):
        events.append(name)
        if observer.time is None:
            observer.set_time(UTC_TIME)
            observer.set_location(52.0, 19.0, 100.0)
            observer.set_look_direction([1.0, 0.0, -1.0], [0.0, 0.0, -1.0])
        if camera.geometry.width is None:
            camera.set_image(16, 12, ImageFormat.MONO8)
            camera.set_fov(40.0)
        return observer, camera

    step.__name__ = name
    return step


class OtsuTests(unittest.TestCase):
    def test_exact_bimodal_threshold_is_independent_of_bit_depth(self):
        mono8 = np.array([10] * 90 + [200] * 10, dtype=np.uint8)
        mono16 = np.array([1000] * 90 + [50000] * 10, dtype=np.uint16)

        self.assertEqual(otsu_threshold(mono8), 105.0)
        self.assertEqual(otsu_threshold(mono16), 25500.0)

    def test_non_finite_values_are_ignored(self):
        values = np.array([0.0, 0.0, 10.0, np.nan, np.inf])
        self.assertEqual(otsu_threshold(values), 5.0)

    def test_constant_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "two distinct"):
            otsu_threshold(np.ones((4, 4)))


class CalibrationPipelineTests(unittest.TestCase):
    def test_custom_steps_run_in_order_and_return_same_pair_contract(self):
        events = []
        steps = [
            configure_pair(events, "configure"),
            CalibrationStep.from_callable(
                "finish",
                configure_pair(events, "finish"),
            ),
        ]
        reports = []
        calibrator = Calibrator(
            empty_sky(),
            steps=steps,
            diagnostic_callback=lambda report, _observer, _camera: reports.append(
                report.name
            ),
        )

        observer, camera = calibrator.estimate_camera(
            Image.new("L", (16, 12)),
            UTC_TIME,
            52.0,
            19.0,
        )

        self.assertEqual(events, ["configure", "finish"])
        self.assertEqual(reports, ["configure", "finish"])
        self.assertEqual(
            [report.name for report in calibrator.last_reports],
            ["configure", "finish"],
        )
        self.assertIsNotNone(observer.observer_matrix)
        self.assertTrue(camera.is_valid())

    def test_plain_callable_constructor_steps_receive_automatic_names(self):
        def configure(observer, camera):
            return configure_pair([], "unused")(observer, camera)

        calibrator = Calibrator(empty_sky(), steps=[configure])

        self.assertEqual([step.name for step in calibrator.steps], ["configure"])

    def test_visualization_callback_receives_only_the_final_effect(self):
        events = []
        visual_reports = []

        def final_binder(context):
            finish = configure_pair(events, "finish")

            def final_step(observer, camera):
                observer, camera = finish(observer, camera)
                context.set_visual(Image.new("RGB", (8, 6)))
                return observer, camera

            return final_step

        calibrator = Calibrator(
            empty_sky(),
            steps=[
                configure_pair(events, "configure"),
                CalibrationStep("finish", final_binder),
            ],
            visualization_callback=visual_reports.append,
        )

        calibrator.estimate_camera(
            Image.new("L", (16, 12)),
            UTC_TIME,
            52.0,
            19.0,
        )

        self.assertEqual(events, ["configure", "finish"])
        self.assertEqual([report.name for report in visual_reports], ["finish"])
        self.assertIsNone(calibrator.last_reports[0].visual)
        self.assertIsInstance(calibrator.last_reports[-1].visual, Image.Image)

    def test_steps_can_be_inserted_added_and_removed(self):
        events = []
        calibrator = Calibrator(
            empty_sky(),
            steps=[
                CalibrationStep.from_callable(
                    "configure",
                    configure_pair(events, "configure"),
                )
            ],
        )
        calibrator.add_step("last", configure_pair(events, "last"))
        calibrator.insert_step(1, "middle", configure_pair(events, "middle"))
        removed = calibrator.remove_step("last")
        self.assertEqual(removed.name, "last")
        calibrator.add_step_definition(
            CalibrationStep.from_callable(
                "last_again",
                configure_pair(events, "last_again"),
            )
        )

        calibrator.estimate_camera(
            Image.new("L", (16, 12)),
            UTC_TIME,
            52.0,
            19.0,
        )

        self.assertEqual(events, ["configure", "middle", "last_again"])

    def test_duplicate_step_name_is_rejected(self):
        step = CalibrationStep.from_callable(
            "same",
            lambda observer, camera: (observer, camera),
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            Calibrator(empty_sky(), steps=[step, step])

    def test_invalid_step_result_has_step_name(self):
        calibrator = Calibrator(
            empty_sky(),
            steps=[
                CalibrationStep.from_callable(
                    "broken",
                    lambda _observer, _camera: None,
                )
            ],
        )

        with self.assertRaisesRegex(CalibrationError, r"broken:"):
            calibrator.estimate_camera(
                Image.new("L", (4, 4)),
                UTC_TIME,
                0.0,
                0.0,
            )


class ImagePropertiesStepTests(unittest.TestCase):
    def make_context(self, image, time=UTC_TIME):
        sky = empty_sky()
        config = CalibrationConfig(exposure_time=0.25)
        return CalibrationContext(
            sky=sky,
            image=image,
            time=time,
            latitude_deg=52.0,
            longitude_deg=19.0,
            elevation_m=100.0,
            config=config,
        )

    def run_first_step(self, context):
        calibrator = Calibrator(context.sky, config=context.config)
        observer = Observer()
        camera = Camera()
        return calibrator.steps[0].bind(context)(observer, camera)

    def test_native_16_bit_data_is_preserved_and_camera_is_populated(self):
        data = np.array([[100, 200], [300, 400]], dtype=np.uint16)
        context = self.make_context(Image.fromarray(data))

        observer, camera = self.run_first_step(context)

        self.assertEqual(
            (camera.geometry.width, camera.geometry.height),
            (2, 2),
        )
        self.assertIs(camera.geometry.image_format, ImageFormat.MONO16)
        self.assertEqual(camera.image_model.exposure_time, 0.25)
        self.assertEqual(camera.image_model.sky_background, 0.0)
        self.assertEqual(camera.image_model.read_noise, 0.0)
        np.testing.assert_array_equal(context.native_data, data)
        np.testing.assert_array_equal(context.image_data, data.astype(float))
        self.assertEqual(context.navigation_image.mode, "L")
        np.testing.assert_array_equal(
            np.asarray(context.navigation_image),
            (data / 257).astype(np.uint8),
        )
        self.assertEqual(
            observer.time,
            UTC_TIME + datetime.timedelta(seconds=0.25),
        )

    def test_intermediate_step_has_no_visualization_when_enabled(self):
        context = self.make_context(Image.new("L", (8, 6), color=12))
        context.visualization_enabled = True

        self.run_first_step(context)
        report = context.finish_report(Calibrator.IMAGE_PROPERTIES_STEP)

        self.assertIsNone(report.visual)

    def test_unsupported_mode_and_naive_time_are_rejected(self):
        palette_context = self.make_context(Image.new("P", (4, 4)))
        with self.assertRaisesRegex(ValueError, "unsupported image mode"):
            self.run_first_step(palette_context)

        naive_context = self.make_context(
            Image.new("L", (4, 4)),
            time=UTC_TIME.replace(tzinfo=None),
        )
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            self.run_first_step(naive_context)

    def test_timestamp_reference_is_converted_to_renderer_exposure_end(self):
        duration = datetime.timedelta(seconds=2.0)
        expected = {
            "start": UTC_TIME + duration,
            "midpoint": UTC_TIME + duration / 2,
            "end": UTC_TIME,
        }
        for reference, expected_end in expected.items():
            with self.subTest(reference=reference):
                config = CalibrationConfig(
                    exposure_time=duration.total_seconds(),
                    time_reference=reference,
                )
                context = CalibrationContext(
                    sky=empty_sky(),
                    image=Image.new("L", (4, 4)),
                    time=UTC_TIME,
                    latitude_deg=52.0,
                    longitude_deg=19.0,
                    elevation_m=0.0,
                    config=config,
                )
                calibrator = Calibrator(context.sky, config=config)

                observer, _camera = calibrator.steps[0].bind(context)(
                    Observer(),
                    Camera(),
                )

                self.assertEqual(observer.time, expected_end)


class PhotometryFitTests(unittest.TestCase):
    def test_shared_moffat_fwhm_and_flux_are_recovered(self):
        calibrator = Calibrator(empty_sky())
        camera = Camera(
            geometry=CameraGeometry(
                fov=40.0,
                width=41,
                height=41,
                image_format=ImageFormat.MONO8,
            ),
            image_model=CameraImageModel(exposure_time=0.5),
        )
        true_flux = 25000.0
        true_fwhm = 3.4
        background = 20.0
        measurements = []
        yy, xx = np.indices((41, 41))
        fit_mask = (xx - 20.2) ** 2 + (yy - 19.7) ** 2 <= 12 ** 2

        for index, magnitude in enumerate((1.0, 2.0, 3.0, 4.0)):
            psf = calibrator._point_spread_function(
                (41, 41),
                centroid_y=19.7,
                centroid_x=20.2,
                fwhm=true_fwhm,
            )
            signal = (
                camera.image_model.exposure_time
                * true_flux
                * 10 ** (-0.4 * magnitude)
            )
            patch = background + signal * psf
            measurements.append(StarMeasurement(
                catalog_id=index,
                magnitude=magnitude,
                catalog_x=20.2,
                catalog_y=19.7,
                centroid_x=20.2,
                centroid_y=19.7,
                area=20,
                background=background,
                background_sigma=1.0,
                aperture_sum=signal,
                patch=patch,
                source_mask=patch > background + 1.0,
                fit_mask=fit_mask,
                patch_x0=0,
                patch_y0=0,
            ))

        flux, fwhm = calibrator._fit_star_photometry(
            measurements,
            camera,
            calibrator.config,
        )

        self.assertAlmostEqual(flux, true_flux, delta=true_flux * 1e-4)
        self.assertAlmostEqual(fwhm, true_fwhm, delta=1e-4)


class DistortionGateTests(unittest.TestCase):
    def test_distortion_is_measured_in_pixels(self):
        camera = Camera(
            geometry=CameraGeometry(
                fov=10.0,
                width=1400,
                height=1000,
            ),
        )
        solution = LostInSpace.Solution(
            ra=0.0,
            dec=0.0,
            roll=0.0,
            fov=10.0,
            distortion=-0.001,
            rmse=1.0,
            matches=6,
            false_positive_prob=1e-9,
            visual=Image.new("RGB", (1, 1)),
            matched_centroids=((500.0, 700.0),) * 6,
            matched_stars=((0.0, 0.0, 1.0),) * 6,
            matched_catalog_ids=tuple(range(6)),
        )

        displacement = Calibrator._distortion_displacement_px(camera, solution)

        self.assertGreater(displacement, 0.0)
        self.assertLess(displacement, 1.0)


class DeferredMatchProcessingTests(unittest.TestCase):
    @staticmethod
    def make_solution(centroids, stars, catalog_ids):
        return LostInSpace.Solution(
            ra=10.0,
            dec=20.0,
            roll=30.0,
            fov=10.0,
            distortion=0.0,
            rmse=1.0,
            matches=6,
            false_positive_prob=1e-9,
            visual=Image.new("RGB", (1, 1)),
            matched_centroids=centroids,
            matched_stars=stars,
            matched_catalog_ids=catalog_ids,
        )

    def test_calibrator_converts_and_aligns_raw_tetra3_payloads(self):
        raw_centroids = [
            [100.0 + index, 200.0 + index]
            for index in range(6)
        ]
        raw_stars = np.asarray([
            [20.0 + index, 30.0 + index, 2.0 + index]
            for index in range(6)
        ])
        raw_catalog_ids = np.asarray([
            [1000 + index, 2000 + index]
            for index in range(6)
        ], dtype=np.int64)
        solution = self.make_solution(
            raw_centroids,
            raw_stars,
            raw_catalog_ids,
        )
        camera = Camera(
            geometry=CameraGeometry(
                fov=10.0,
                width=1400,
                height=1000,
            ),
        )
        calibrator = Calibrator(empty_sky())

        (
            _rmse_px,
            _distortion_px,
            centroids,
            stars,
            catalog_ids,
        ) = calibrator._validate_refined_solution(
            camera,
            solution,
            calibrator.config,
        )

        self.assertIs(solution.matched_centroids, raw_centroids)
        self.assertIs(solution.matched_stars, raw_stars)
        self.assertIs(solution.matched_catalog_ids, raw_catalog_ids)
        self.assertEqual(centroids.shape, (6, 2))
        self.assertEqual(stars.shape, (6, 3))
        self.assertTrue(np.issubdtype(centroids.dtype, np.floating))
        self.assertEqual(
            catalog_ids,
            tuple(
                (1000 + index, 2000 + index)
                for index in range(6)
            ),
        )

    def test_calibrator_rejects_malformed_raw_match_shape(self):
        solution = self.make_solution(
            centroids=[1.0, 2.0, 3.0],
            stars=np.ones((6, 3)),
            catalog_ids=np.arange(6),
        )

        with self.assertRaisesRegex(
            ValueError,
            r"matched_centroids must have shape",
        ):
            Calibrator._process_matches(solution)

    def test_calibrator_rejects_inconsistent_reported_match_count(self):
        solution = self.make_solution(
            centroids=np.ones((5, 2)),
            stars=np.ones((5, 3)),
            catalog_ids=np.arange(5),
        )

        with self.assertRaisesRegex(ValueError, r"match count"):
            Calibrator._process_matches(solution)


class CalibrationVisualizationTests(unittest.TestCase):
    def test_visualizer_builds_only_the_final_comparison(self):
        original = Image.fromarray(
            np.arange(96, dtype=np.uint8).reshape(8, 12)
        )
        rendered = Image.new("L", original.size, 20)
        frame = CalibrationVisualizer.final_result(
            original,
            rendered,
            {"fov_deg": 10.0, "fwhm_px": 3.0},
        )

        self.assertIsInstance(frame, Image.Image)
        self.assertEqual(frame.mode, "RGB")
        self.assertGreater(frame.width, original.width)
        self.assertGreater(frame.height, original.height)

        limits = CalibrationVisualizer.stretch_limits(original)
        expected_render = CalibrationVisualizer.preview(rendered, limits)
        body_y = (
            CalibrationVisualizer.HEADER_HEIGHT
            + CalibrationVisualizer.LABEL_HEIGHT
        )
        rendered_panel = frame.crop((
            original.width,
            body_y,
            original.width + rendered.width,
            body_y + rendered.height,
        ))
        np.testing.assert_array_equal(
            np.asarray(rendered_panel),
            np.asarray(expected_render),
        )


class SkyNoiseStepTests(unittest.TestCase):
    def test_blank_sky_step_sets_camera_noise_and_reports_identifiability(self):
        rng = np.random.default_rng(123)
        data = rng.poisson(20.0, size=(64, 64)).astype(np.uint8)
        image = Image.fromarray(data)
        config = CalibrationConfig(
            sky_tile_size=16,
            maximum_noise_samples=5000,
        )
        context = CalibrationContext(
            sky=empty_sky(),
            image=image,
            time=UTC_TIME,
            latitude_deg=0.0,
            longitude_deg=0.0,
            elevation_m=0.0,
            config=config,
            image_data=data.astype(float),
            native_data=data,
            saturated_mask=np.zeros(data.shape, dtype=bool),
        )
        observer = Observer()
        observer.set_time(UTC_TIME + datetime.timedelta(seconds=1))
        observer.set_location(0.0, 0.0)
        observer.set_look_direction(
            look_dir=[0.0, 0.0, -1.0],
            look_up=[1.0, 0.0, 0.0],
        )
        camera = Camera(
            geometry=CameraGeometry(
                fov=30.0,
                width=64,
                height=64,
                image_format=ImageFormat.MONO8,
            ),
            image_model=CameraImageModel(
                exposure_time=1.0,
                fwhm=3.0,
            ),
        )
        calibrator = Calibrator(context.sky, config=config)

        with self.assertWarnsRegex(RuntimeWarning, "Poisson and quantization"):
            returned_observer, returned_camera = calibrator.steps[-1].bind(
                context
            )(
                observer,
                camera,
            )
        report = context.finish_report(calibrator.SKY_NOISE_STEP)

        self.assertIs(returned_observer, observer)
        self.assertIs(returned_camera, camera)
        self.assertAlmostEqual(
            camera.image_model.sky_background,
            20.0,
            delta=1.0,
        )
        self.assertGreaterEqual(camera.image_model.read_noise, 0.0)
        self.assertIn("read_noise_identifiable", report.values)
        self.assertFalse(report.values["read_noise_identifiable"])

    def test_mono16_background_and_read_noise_are_recovered_in_native_counts(
        self,
    ):
        rng = np.random.default_rng(987)
        expected_background = 350.0
        expected_read_noise = 8.0
        data = np.rint(
            rng.poisson(expected_background, size=(256, 256))
            + rng.normal(0.0, expected_read_noise, size=(256, 256))
        )
        data = np.clip(data, 0, 65535).astype(np.uint16)
        image = Image.fromarray(data)
        config = CalibrationConfig(
            sky_tile_size=32,
            maximum_noise_samples=100000,
        )
        context = CalibrationContext(
            sky=empty_sky(),
            image=image,
            time=UTC_TIME,
            latitude_deg=0.0,
            longitude_deg=0.0,
            elevation_m=0.0,
            config=config,
            image_data=data.astype(float),
            native_data=data,
            saturated_mask=np.zeros(data.shape, dtype=bool),
        )
        observer = Observer()
        observer.set_time(UTC_TIME + datetime.timedelta(seconds=1))
        observer.set_location(0.0, 0.0)
        observer.set_look_direction(
            look_dir=[0.0, 0.0, -1.0],
            look_up=[1.0, 0.0, 0.0],
        )
        camera = Camera(
            geometry=CameraGeometry(
                fov=30.0,
                width=256,
                height=256,
                image_format=ImageFormat.MONO16,
            ),
            image_model=CameraImageModel(
                exposure_time=1.0,
                fwhm=3.0,
            ),
        )
        calibrator = Calibrator(context.sky, config=config)

        calibrator.steps[-1].bind(context)(observer, camera)

        self.assertAlmostEqual(
            camera.image_model.sky_background,
            expected_background,
            delta=2.0,
        )
        self.assertAlmostEqual(
            camera.image_model.read_noise,
            expected_read_noise,
            delta=1.0,
        )
        self.assertTrue(context._pending_values["read_noise_identifiable"])


if __name__ == "__main__":
    unittest.main()
