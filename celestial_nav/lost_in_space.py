from dataclasses import dataclass
from numbers import Integral
from PIL import Image
from pathlib import Path
from shutil import copy2
from typing import Any, Optional
import tetra3

from common import Config


def _validate_pattern_checking_stars(value: object) -> Optional[int]:
    """Return a valid tetra3 pattern-search depth."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("pattern_checking_stars must be an integer or None")
    if value < 4:
        raise ValueError("pattern_checking_stars must be at least 4")
    return int(value)


class LostInSpace:

    @dataclass
    class Solution:
        ra: float
        dec: float
        roll: float

        fov: float
        distortion: float
        rmse: float
        matches: int
        false_positive_prob: float

        visual: Image.Image

        # Match payloads are intentionally kept exactly as tetra3 returned them.
        matched_centroids: Any = None
        matched_stars: Any = None
        matched_catalog_ids: Any = None

    @staticmethod
    def get_db_path(min_fov, max_fov, star_max_magnitude):
        return Config.CACHE_DIR / f"t3_{int(min_fov)}_{int(max_fov)}_{int(star_max_magnitude)}.npz"

    @staticmethod
    def generate_db(min_fov=40, max_fov=60, star_max_magnitude=7) -> Path:
        package_data = Path(tetra3.__file__).resolve().parent / "data"
        target = package_data / "hip_main.dat"
        source = Config.CACHE_DIR / "hip_main.dat"

        if not target.exists():
            package_data.mkdir(parents=True, exist_ok=True)
            copy2(source, target)

        db_path = LostInSpace.get_db_path(min_fov, max_fov, star_max_magnitude)
        t3 = tetra3.Tetra3(load_database=None)
        t3.generate_database(
            save_as=db_path,
            max_fov=max_fov,
            min_fov=min_fov,
            star_max_magnitude=star_max_magnitude,
            star_catalog="hip_main",
        )

        return db_path
    
    def __init__(self, db_path):
        self._t3 = tetra3.Tetra3(load_database=db_path)

    def solve(
        self,
        image: Image.Image,
        distortion=(-0.5, 0.5),
        pattern_checking_stars = 12,
    ) -> Optional[Solution]:
        _validate_pattern_checking_stars(pattern_checking_stars)
        solve_options = {
            "distortion": distortion,
            "pattern_checking_stars": pattern_checking_stars,
            "return_matches": True,
            "return_visual": True,
        }
        res = self._t3.solve_from_image(image, **solve_options)

        if (
            not res
            or res.get("RA") is None
            or res.get("Dec") is None
            or res.get("Roll") is None
        ):
            return None

        solution = LostInSpace.Solution(
            ra=res["RA"],
            dec=res["Dec"],
            roll=res["Roll"],

            fov=res["FOV"],
            distortion=res["distortion"],
            rmse=res["RMSE"],
            matches=res["Matches"],
            false_positive_prob=res["Prob"],
            visual=res["visual"],
            matched_centroids=res.get("matched_centroids"),
            matched_stars=res.get("matched_stars"),
            matched_catalog_ids=res.get("matched_catID"),
        )
        return solution
