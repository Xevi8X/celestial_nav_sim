import datetime
import math
from typing import Union

import numpy as np
from astropy.visualization import ImageNormalize, SqrtStretch, ZScaleInterval
from PIL import Image, ImageOps

from celestial_nav import LostInSpace, Navigator
from common import Observer, Sky
from sky_render import Camera, ImageFormat, Renderer


class Calibrator:
    FOV_TOLERANCE = 2

    def __init__(self, sky: Sky):
        self.sky = sky

    def _determine_mode(self, image: Image.Image) -> ImageFormat:
        if image.mode == "L":
            return ImageFormat.MONO8
        elif image.mode == "I":
            return ImageFormat.MONO16
        elif image.mode == "RGB":
            return ImageFormat.RGB8

    def estimate_camera(
        self,
        image: Image.Image,
        time: datetime.datetime,
        latitude_deg: float,
        longitude_deg: float,
        elevation_m: float = 0.0,
        exposure_time: float = 1.0,
    ) -> Union[tuple[Observer, Camera], None]:

        lost_in_space = LostInSpace(LostInSpace.get_db_path(min_fov=5, max_fov=120, star_max_magnitude=7))
        rough_solution = lost_in_space.solve(image, distortion=[-0.3, 0.3])

        if rough_solution is None:
            print("Could not find rough camera solution")
            return None

        print(f"Rough solution: FOV={rough_solution.fov}, Distortion={rough_solution.distortion}")

        tol = self.FOV_TOLERANCE
        rounded_fov = round(rough_solution.fov / tol) * tol
        navigator = Navigator(
            self.sky,
            fov_range=(rounded_fov - tol, rounded_fov + tol),
            star_max_magnitude=10
        )
        solution = navigator.find_solution(image)

        if solution is None:
            print("Could not find refined camera solution")
            return None

        print(f"Refined solution: FOV={solution.fov}, Distortion={solution.distortion}")

        camera = Camera(
            width=image.width,
            height=image.height,
            image_format=self._determine_mode(image),

            fov=solution.fov,
            exposure_time=exposure_time,

            flux=1e5,
            sky_background=0.1,
            read_noise=0.1,
            fwhm=5.0,
        )

        observer = navigator.estimate_orientation(
            image,
            time=time,
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            elevation_m=elevation_m,
        )

        renderer = Renderer(self.sky, camera)

        # original_data = np.asarray(image, dtype=np.float32)
        # normalize = ImageNormalize(
        #     original_data,
        #     interval=ZScaleInterval(),
        #     stretch=SqrtStretch(),
        #     clip=True,
        # )

        # original_data = normalize(original_data) * 255
        original_data = np.asarray(image, dtype=np.float32) // 257
        original = ImageOps.invert(
            Image.fromarray(np.asarray(original_data, dtype=np.uint8)).convert("L")
        )

        rendered = ImageOps.invert(renderer.render(observer).convert("L"))

        combined_width = original.width + rendered.width
        combined_height = max(original.height, rendered.height)
        combined = Image.new("L", (combined_width, combined_height))
        combined.paste(original, (0, 0))
        combined.paste(rendered, (original.width, 0))
        combined.show()

        return observer, camera
