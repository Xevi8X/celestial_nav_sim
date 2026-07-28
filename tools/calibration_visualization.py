from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


Panel = Tuple[str, Image.Image]


class CalibrationVisualizer:
    """Build the final observed-versus-calibrated comparison."""

    MAX_PANEL_WIDTH = 800
    HEADER_HEIGHT = 52
    LABEL_HEIGHT = 24

    @staticmethod
    def _luminance(data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 3:
            return np.average(
                data[:, :, :3],
                axis=2,
                weights=(0.299, 0.587, 0.114),
            )
        return data

    @classmethod
    def stretch_limits(cls, image: Image.Image) -> Tuple[float, float]:
        data = cls._luminance(np.asarray(image))
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return 0.0, 1.0
        low, high = np.percentile(finite, (1.0, 99.9))
        if not high > low:
            low = float(np.min(finite))
            high = float(np.max(finite))
        if not high > low:
            high = low + 1.0
        return float(low), float(high)

    @classmethod
    def preview(
        cls,
        image: Image.Image,
        limits: Optional[Tuple[float, float]] = None,
    ) -> Image.Image:
        data = cls._luminance(np.asarray(image))
        low, high = cls.stretch_limits(image) if limits is None else limits
        normalized = np.clip((data - low) / (high - low), 0, 1)
        output = np.asarray(np.sqrt(normalized) * 255, dtype=np.uint8)
        return Image.fromarray(output).convert("RGB")

    @classmethod
    def _fit_panel(cls, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if image.width <= cls.MAX_PANEL_WIDTH:
            return image
        height = max(
            1,
            round(image.height * cls.MAX_PANEL_WIDTH / image.width),
        )
        return image.resize(
            (cls.MAX_PANEL_WIDTH, height),
            Image.Resampling.LANCZOS,
        )

    @staticmethod
    def _metrics_text(values: Mapping[str, object]) -> str:
        items = []
        for key, value in values.items():
            if isinstance(value, float):
                rendered = f"{value:.4g}"
            else:
                rendered = str(value)
            items.append(f"{key}={rendered}")
            if len(items) == 6:
                break
        return "   ".join(items)

    @classmethod
    def compose(
        cls,
        title: str,
        panels: Sequence[Panel],
        values: Optional[Mapping[str, object]] = None,
    ) -> Image.Image:
        fitted = [
            (label, cls._fit_panel(image))
            for label, image in panels
        ]
        body_height = max(image.height for _label, image in fitted)
        width = sum(image.width for _label, image in fitted)
        height = cls.HEADER_HEIGHT + cls.LABEL_HEIGHT + body_height
        output = Image.new("RGB", (width, height), (18, 18, 22))
        draw = ImageDraw.Draw(output)
        draw.text((10, 7), title, fill=(255, 255, 255))
        if values:
            draw.text(
                (10, 28),
                cls._metrics_text(values),
                fill=(185, 205, 225),
            )

        x = 0
        for label_text, image in fitted:
            label_y = cls.HEADER_HEIGHT
            draw.rectangle(
                (x, label_y, x + image.width, label_y + cls.LABEL_HEIGHT),
                fill=(35, 38, 45),
            )
            draw.text((x + 8, label_y + 5), label_text, fill=(235, 235, 235))
            output.paste(image, (x, label_y + cls.LABEL_HEIGHT))
            x += image.width
        return output

    @classmethod
    def final_result(
        cls,
        original: Image.Image,
        rendered: Image.Image,
        values: Mapping[str, object],
    ) -> Image.Image:
        limits = cls.stretch_limits(original)
        return cls.compose(
            "Calibration result",
            (
                ("observed image", cls.preview(original, limits)),
                ("calibrated camera render", cls.preview(rendered, limits)),
            ),
            values,
        )
