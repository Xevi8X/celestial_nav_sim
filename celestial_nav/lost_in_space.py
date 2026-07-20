from dataclasses import dataclass

from PIL import Image
from pathlib import Path
from shutil import copy2
import tetra3

from common import Config

class LostInSpace:

    @dataclass
    class Solution:
        ra: float
        dec: float
        roll: float

        fov: float
        distortion: float
        false_positive_prob: float

        visual: Image.Image

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
        self._distortion = [-0.05, 0.05]

    def solve(self, image: Image.Image):
        res = self._t3.solve_from_image(
            image, 
            distortion=self._distortion,
            return_visual=True
            )
        Solution = LostInSpace.Solution(
            ra=res["RA"],
            dec=res["Dec"],
            roll=res["Roll"],
            fov=res["FOV"],
            distortion=res["distortion"],
            false_positive_prob=res["Prob"],
            visual=res["visual"],
        )

        return Solution
