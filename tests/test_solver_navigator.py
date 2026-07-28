import datetime
import inspect
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from celestial_nav import LostInSpace, Navigator


def solver_result(**overrides):
    result = {
        "RA": 123.5,
        "Dec": -22.25,
        "Roll": 17.0,
        "FOV": 42.0,
        "distortion": -0.01,
        "RMSE": 3.5,
        "Matches": 2,
        "Prob": 1e-8,
        "visual": Image.new("RGB", (8, 6)),
        "matched_centroids": [[1.25, 2.5], [3.75, 4.0]],
        "matched_stars": [[10.0, 20.0, 1.5], [30.0, 40.0, 2.5]],
        "matched_catID": [101, 202],
    }
    result.update(overrides)
    return result


def solution():
    return LostInSpace.Solution(
        ra=123.5,
        dec=-22.25,
        roll=17.0,
        fov=42.0,
        distortion=-0.01,
        rmse=3.5,
        matches=2,
        false_positive_prob=1e-8,
        visual=Image.new("RGB", (8, 6)),
    )


class LostInSpaceMatchTests(unittest.TestCase):
    def make_solver(self, result):
        tetra3_solver = Mock()
        tetra3_solver.solve_from_image.return_value = result
        with patch(
            "celestial_nav.lost_in_space.tetra3.Tetra3",
            return_value=tetra3_solver,
        ):
            lost_in_space = LostInSpace("unused.npz")
        return lost_in_space, tetra3_solver

    def test_solve_requests_and_exposes_raw_match_data(self):
        raw_result = solver_result()
        lost_in_space, tetra3_solver = self.make_solver(raw_result)
        image = Image.new("L", (8, 6))

        result = lost_in_space.solve(image)

        tetra3_solver.solve_from_image.assert_called_once_with(
            image,
            distortion=(-0.1, 0.1),
            return_matches=True,
            return_visual=True,
        )
        self.assertIs(
            result.matched_centroids,
            raw_result["matched_centroids"],
        )
        self.assertIs(
            result.matched_stars,
            raw_result["matched_stars"],
        )
        self.assertIs(
            result.matched_catalog_ids,
            raw_result["matched_catID"],
        )

    def test_solve_preserves_numpy_catalog_ids_without_processing(self):
        raw_result = solver_result(
            matched_catID=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
        )
        lost_in_space, _ = self.make_solver(raw_result)

        result = lost_in_space.solve(Image.new("L", (8, 6)))

        self.assertIs(
            result.matched_catalog_ids,
            raw_result["matched_catID"],
        )

    def test_solve_allows_database_without_catalog_ids(self):
        lost_in_space, _ = self.make_solver(solver_result(matched_catID=None))

        result = lost_in_space.solve(Image.new("L", (8, 6)))

        self.assertIsNone(result.matched_catalog_ids)

    def test_solve_leaves_misaligned_correspondences_for_caller(self):
        raw_result = solver_result(
            matched_stars=[[10.0, 20.0, 1.5]],
        )
        lost_in_space, _ = self.make_solver(raw_result)

        result = lost_in_space.solve(Image.new("L", (8, 6)))

        self.assertIs(
            result.matched_centroids,
            raw_result["matched_centroids"],
        )
        self.assertIs(result.matched_stars, raw_result["matched_stars"])

    def test_unsolved_image_returns_none_without_requiring_match_keys(self):
        lost_in_space, _ = self.make_solver(
            {
                "RA": None,
                "Dec": None,
                "Roll": None,
            }
        )

        self.assertIsNone(lost_in_space.solve(Image.new("L", (8, 6))))


class NavigatorOrientationTests(unittest.TestCase):
    def make_navigator(self):
        navigator = Navigator.__new__(Navigator)
        navigator.sky = Mock()
        navigator.sky.radec_to_ecef.return_value = (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )
        navigator.find_solution = Mock()
        return navigator

    def test_estimate_orientation_has_no_solution_handoff(self):
        parameters = inspect.signature(
            Navigator.estimate_orientation,
        ).parameters

        self.assertNotIn("solution", parameters)

    def test_estimate_orientation_solves_image_once(self):
        navigator = self.make_navigator()
        known_solution = solution()
        navigator.find_solution.return_value = known_solution
        image = Image.new("L", (8, 6))

        with patch(
            "celestial_nav.navigator.ECEF.ecef_to_ned",
            return_value=np.eye(3),
        ):
            observer = navigator.estimate_orientation(
                image,
                time=datetime.datetime(
                    2026,
                    7,
                    28,
                    tzinfo=datetime.timezone.utc,
                ),
                latitude_deg=51.0,
                longitude_deg=19.0,
            )

        navigator.find_solution.assert_called_once_with(image)
        self.assertIsNotNone(observer)

    def test_failed_solve_returns_initialized_observer(self):
        navigator = self.make_navigator()
        navigator.find_solution.return_value = None

        image = Image.new("L", (8, 6))
        timestamp = datetime.datetime(
            2026,
            7,
            28,
            tzinfo=datetime.timezone.utc,
        )
        observer = navigator.estimate_orientation(
            image,
            time=timestamp,
            latitude_deg=51.0,
            longitude_deg=19.0,
        )

        self.assertEqual(observer.time, timestamp)
        self.assertEqual(observer.latitude, 51.0)
        self.assertEqual(observer.longitude, 19.0)
        self.assertIsNone(observer.observer_matrix)
        navigator.find_solution.assert_called_once_with(image)


if __name__ == "__main__":
    unittest.main()
