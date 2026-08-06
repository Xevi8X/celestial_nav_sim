"""Deterministic reporting helpers for the article validation experiments.

The scientific code deliberately does not depend on this module.  Reporting
functions accept ordinary sequences of mappings, which keeps simulations,
record persistence, and publication formatting separate.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import io
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont


ARTIFACT_FILENAMES = MappingProxyType(
    {
        "simulation_locations": "simulation_locations.csv",
        "renderer_tuning_results": "renderer_tuning_results.csv",
        "accelerometer_results": "accelerometer_results.csv",
        "magnetometer_results": "magnetometer_results.csv",
        "position_results": "position_results.csv",
        "renderer_image_comparison_png": "renderer_image_comparison.png",
        "renderer_image_comparison_eps": "renderer_image_comparison.eps",
        "renderer_histogram_comparison": "renderer_histogram_comparison.eps",
        "accelerometer_figure": "accelerometer_results.eps",
        "magnetometer_figure": "magnetometer_results.eps",
        "position_figure": "position_results.eps",
    }
)


# Leading columns have a stable, article-oriented order.  Extra columns are
# retained by ``write_artifact_csv`` and appended alphabetically.
CSV_SCHEMAS = MappingProxyType(
    {
        "simulation_locations": (
            "location",
            "latitude_deg",
            "longitude_deg",
        ),
        "renderer_tuning_results": (
            "parameter",
            "image1",
            "image2",
            "image3",
            "average",
            "unit",
        ),
        "accelerometer_results": (
            "location",
            "azimuth_deg",
            "rotation_error_deg",
            "scale_error_percent",
            "bias_error_mps2",
            "calibration_time_s",
            "status",
        ),
        "magnetometer_results": (
            "location",
            "procedure",
            "vector_error_ut",
            "heading_error_deg",
            "test_orientation_count",
            "status",
        ),
        "position_results": (
            "location",
            "latitude_deg",
            "longitude_deg",
            "individual_image_mean_error_m",
            "four_pose_error_m",
            "accepted_images",
            "rejected_images",
            "status",
        ),
    }
)


def _jsonable(value: Any) -> Any:
    """Return a stable JSON-compatible representation of ``value``."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite values cannot be serialized")
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.timezone.utc)
        value = value.astimezone(dt.timezone.utc)
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        items = [_jsonable(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"unsupported value for serialization: {type(value).__name__}")


def _csv_value(value: Any, float_precision: int) -> str:
    value = _jsonable(value)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, f".{float_precision}g")
    if isinstance(value, (list, dict)):
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return str(value)


def _materialize_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"record {index} is not a mapping")
        row: dict[str, Any] = {}
        for key, value in record.items():
            if not isinstance(key, str):
                raise TypeError(f"record {index} contains a non-string field name")
            row[key] = value
        rows.append(row)
    return rows


def write_records_csv(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
    sort_by: str | Sequence[str] | None = None,
    float_precision: int = 12,
) -> Path:
    """Write records with canonical newlines, values, and column ordering.

    Input order is retained unless ``sort_by`` is supplied.  When field names
    are omitted, the union of all record fields is sorted alphabetically.
    """

    if float_precision < 1:
        raise ValueError("float_precision must be positive")
    rows = _materialize_records(records)
    if fieldnames is None:
        fields = sorted({key for row in rows for key in row})
        if not fields:
            raise ValueError("fieldnames are required when no records are supplied")
    else:
        fields = list(fieldnames)
        if not fields or len(fields) != len(set(fields)):
            raise ValueError("fieldnames must be non-empty and unique")
        extras = sorted({key for row in rows for key in row}.difference(fields))
        if extras:
            raise ValueError(f"fields absent from fieldnames: {', '.join(extras)}")

    if sort_by is not None:
        sort_fields = [sort_by] if isinstance(sort_by, str) else list(sort_by)
        unknown = set(sort_fields).difference(fields)
        if unknown:
            raise ValueError(f"unknown sort fields: {', '.join(sorted(unknown))}")

        def sort_key(row: Mapping[str, Any]) -> tuple[str, ...]:
            return tuple(_csv_value(row.get(field), float_precision) for field in sort_fields)

        rows.sort(key=sort_key)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _csv_value(row.get(field), float_precision)
                    for field in fields
                }
            )
    return destination


def write_artifact_csv(
    output_directory: str | Path,
    artifact: str,
    records: Iterable[Mapping[str, Any]],
    *,
    sort_by: str | Sequence[str] | None = None,
    float_precision: int = 12,
) -> Path:
    """Write one of the named article CSV files using its stable schema."""

    if artifact not in CSV_SCHEMAS:
        raise KeyError(f"unknown CSV artifact: {artifact}")
    rows = _materialize_records(records)
    leading_fields = CSV_SCHEMAS[artifact]
    extras = sorted({key for row in rows for key in row}.difference(leading_fields))
    fields = (*leading_fields, *extras)
    path = Path(output_directory) / ARTIFACT_FILENAMES[artifact]
    return write_records_csv(
        path,
        rows,
        fieldnames=fields,
        sort_by=sort_by,
        float_precision=float_precision,
    )


def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    unique: list[Any] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _require_fields(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    for index, row in enumerate(rows):
        missing = [field for field in fields if field not in row]
        if missing:
            raise ValueError(f"record {index} is missing fields: {', '.join(missing)}")


def _finite_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(rows: Iterable[Mapping[str, Any]], field: str) -> float:
    values = [number for row in rows if (number := _finite_number(row.get(field))) is not None]
    return float(np.mean(values)) if values else math.nan


def _save_vector_eps(figure: Figure, path: str | Path) -> Path:
    """Save paired EPS/PNG figures and normalize EPS wall-clock metadata."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    png_destination = destination.with_suffix(".png")
    figure.savefig(
        png_destination,
        format="png",
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "celestial_nav_sim"},
    )
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="eps",
        bbox_inches="tight",
        metadata={"Creator": "celestial_nav_sim"},
    )
    content = buffer.getvalue().decode("latin-1")
    content = re.sub(
        r"^%%CreationDate:.*$",
        "%%CreationDate: 1970-01-01T00:00:00Z",
        content,
        flags=re.MULTILINE,
    )
    content = re.sub(
        r"^%%Title:.*$",
        f"%%Title: {destination.name}",
        content,
        flags=re.MULTILINE,
    )
    destination.write_bytes(content.encode("latin-1"))
    figure.clear()
    return destination


def _legend_below(figure: Figure, axis: Any, title: str | None = None) -> None:
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title=title,
        loc="outside lower center",
        ncols=len(labels),
    )


def _add_grid(axis: Any) -> None:
    axis.set_axisbelow(True)
    axis.grid()


def plot_renderer_histograms(
    records: Iterable[Mapping[str, Any]],
    path: str | Path,
    *,
    image_field: str = "image_id",
    series_field: str = "source",
    intensity_field: str = "intensity",
    value_field: str = "fraction",
) -> Path:
    """Plot real/rendered intensity distributions, one panel per image."""

    rows = _materialize_records(records)
    if not rows:
        raise ValueError("histogram records cannot be empty")
    _require_fields(
        rows,
        (image_field, series_field, intensity_field, value_field),
    )
    image_ids = _ordered_unique(row[image_field] for row in rows)
    series = _ordered_unique(row[series_field] for row in rows)
    figure = Figure(
        figsize=(max(3.4, 2.6 * len(image_ids)), 2.8),
        constrained_layout=True,
    )
    axes = figure.subplots(
        1,
        len(image_ids),
        squeeze=False,
        sharey=True,
    )[0]
    for axis, image_id in zip(axes, image_ids):
        for series_name in series:
            selected = [
                row
                for row in rows
                if row[image_field] == image_id and row[series_field] == series_name
            ]
            selected.sort(key=lambda row: float(row[intensity_field]))
            x_values = [float(row[intensity_field]) for row in selected]
            y_values = [float(row[value_field]) for row in selected]
            if x_values:
                axis.scatter(
                    x_values,
                    y_values,
                    label=str(series_name),
                )
        axis.set_title(str(image_id))
        axis.set_xlabel("Pixel intensity")
        _add_grid(axis)
    axes[0].set_ylabel("Pixel fraction")
    _legend_below(figure, axes[0])
    return _save_vector_eps(figure, path)


_ACCELEROMETER_METRICS = MappingProxyType(
    {
        "rotation_error_deg": "Rotation error (deg)",
        "scale_error_percent": "Scale error (%)",
        "bias_error_mps2": "Bias error (m/s²)",
    }
)


def plot_accelerometer_results(
    records: Iterable[Mapping[str, Any]],
    path: str | Path,
    *,
    error_fields: Mapping[str, str] | None = None,
    location_field: str = "location",
    azimuth_field: str = "azimuth_deg",
) -> Path:
    """Plot calibration errors for every location and azimuth.

    Repeated records for one location/azimuth pair are represented by their
    arithmetic mean.  Missing or non-finite metric values are omitted.
    """

    rows = _materialize_records(records)
    if not rows:
        raise ValueError("accelerometer records cannot be empty")
    _require_fields(rows, (location_field, azimuth_field))
    if error_fields is None:
        metrics = {
            field: label
            for field, label in _ACCELEROMETER_METRICS.items()
            if any(_finite_number(row.get(field)) is not None for row in rows)
        }
    else:
        metrics = dict(error_fields)
    if not metrics:
        raise ValueError("no finite accelerometer metrics were found")

    locations = _ordered_unique(row[location_field] for row in rows)
    azimuths = sorted(
        _ordered_unique(row[azimuth_field] for row in rows),
        key=float,
    )
    figure = Figure(
        figsize=(3.0 * len(metrics), 3.0),
        constrained_layout=True,
    )
    axes = figure.subplots(1, len(metrics), squeeze=False)[0]
    x_values = np.arange(len(locations))
    for axis, (field, label) in zip(axes, metrics.items()):
        for azimuth in azimuths:
            values = []
            for location in locations:
                selected = [
                    row
                    for row in rows
                    if row[location_field] == location
                    and row[azimuth_field] == azimuth
                ]
                values.append(_mean(selected, field))
            axis.scatter(
                x_values,
                values,
                label=f"{float(azimuth):g}°",
            )
        axis.set_ylabel(label)
        axis.set_xticks(x_values, [str(location) for location in locations])
        axis.tick_params(axis="x", labelrotation=20, labelsize=8)
        _add_grid(axis)
    _legend_below(figure, axes[0], title="Initial azimuth")
    return _save_vector_eps(figure, path)


def plot_magnetometer_results(
    records: Iterable[Mapping[str, Any]],
    path: str | Path,
    *,
    location_field: str = "location",
    procedure_field: str = "procedure",
    vector_error_field: str = "vector_error_ut",
    heading_error_field: str = "heading_error_deg",
) -> Path:
    """Plot complete-pattern and constrained-turn precision by location."""

    rows = _materialize_records(records)
    if not rows:
        raise ValueError("magnetometer records cannot be empty")
    _require_fields(
        rows,
        (
            location_field,
            procedure_field,
            vector_error_field,
            heading_error_field,
        ),
    )
    locations = _ordered_unique(row[location_field] for row in rows)
    procedures = _ordered_unique(row[procedure_field] for row in rows)
    figure = Figure(figsize=(7.2, 3.1), constrained_layout=True)
    axes = np.asarray(figure.subplots(1, 2, squeeze=False))[0]
    x_values = np.arange(len(locations), dtype=float)
    width = 0.8 / max(1, len(procedures))
    for axis, field, ylabel in (
        (axes[0], vector_error_field, "Magnetic-vector error (µT)"),
        (axes[1], heading_error_field, "Heading error (deg)"),
    ):
        for index, procedure in enumerate(procedures):
            values = []
            for location in locations:
                selected = [
                    row
                    for row in rows
                    if row[location_field] == location
                    and row[procedure_field] == procedure
                ]
                values.append(_mean(selected, field))
            offsets = x_values - 0.4 + width / 2 + index * width
            axis.bar(
                offsets,
                values,
                width=width,
                label=str(procedure),
            )
        axis.set_ylabel(ylabel)
        axis.set_xticks(x_values, [str(location) for location in locations])
        axis.tick_params(axis="x", labelrotation=20, labelsize=8)
        _add_grid(axis)
    _legend_below(figure, axes[0])
    return _save_vector_eps(figure, path)


def plot_position_results(
    records: Iterable[Mapping[str, Any]],
    path: str | Path,
    *,
    location_field: str = "location",
    individual_field: str = "individual_image_mean_error_m",
    sequence_field: str = "four_pose_error_m",
) -> Path:
    """Plot mean individual-image error against the combined estimate."""

    rows = _materialize_records(records)
    if not rows:
        raise ValueError("position records cannot be empty")
    _require_fields(rows, (location_field, individual_field, sequence_field))
    locations = _ordered_unique(row[location_field] for row in rows)
    x_values = np.arange(len(locations), dtype=float)
    width = 0.36
    individual = [
        _mean((row for row in rows if row[location_field] == location), individual_field)
        for location in locations
    ]
    sequence = [
        _mean((row for row in rows if row[location_field] == location), sequence_field)
        for location in locations
    ]
    figure = Figure(figsize=(7.2, 3.2), constrained_layout=True)
    axis = figure.subplots(1, 1)
    axis.bar(
        x_values - width / 2,
        individual,
        width,
        label="Mean individual-image error",
    )
    axis.bar(
        x_values + width / 2,
        sequence,
        width,
        label="Combined 60 s",
    )
    axis.set_ylabel("Position error (m)")
    axis.set_xticks(x_values, [str(location) for location in locations])
    axis.tick_params(axis="x", labelrotation=20, labelsize=8)
    _add_grid(axis)
    _legend_below(figure, axis)
    return _save_vector_eps(figure, path)


def _load_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.copy()
    if isinstance(value, np.ndarray):
        return Image.fromarray(value)
    path = Path(value)
    with Image.open(path) as image:
        return image.copy()


def _luminance(image: Image.Image) -> np.ndarray:
    data = np.asarray(image, dtype=np.float64)
    if data.ndim == 3:
        data = np.average(data[..., :3], axis=2, weights=(0.299, 0.587, 0.114))
    return data


def _preview_pair(real: Image.Image, rendered: Image.Image) -> tuple[Image.Image, Image.Image]:
    arrays = (_luminance(real), _luminance(rendered))
    finite_parts = [array[np.isfinite(array)] for array in arrays]
    finite = np.concatenate([part for part in finite_parts if part.size])
    if finite.size:
        low, high = np.percentile(finite, (1.0, 99.5))
        if not high > low:
            low, high = float(np.min(finite)), float(np.max(finite))
    else:
        low, high = 0.0, 1.0
    if not high > low:
        high = low + 1.0
    previews = []
    for array in arrays:
        normalized = np.nan_to_num((array - low) / (high - low), nan=0.0)
        output = np.asarray(np.sqrt(np.clip(normalized, 0.0, 1.0)) * 255, dtype=np.uint8)
        previews.append(Image.fromarray(output).convert("RGB"))
    return previews[0], previews[1]


def write_renderer_montage(
    records: Iterable[Mapping[str, Any]],
    png_path: str | Path,
    *,
    eps_path: str | Path | None = None,
    image_field: str = "image_id",
    real_field: str = "real_image",
    rendered_field: str = "rendered_image",
    panel_width: int = 640,
    real_label: str = "Real",
    rendered_label: str = "Rendered",
) -> tuple[Path, Path | None]:
    """Create a raster real/rendered montage and, optionally, an EPS wrapper."""

    rows = _materialize_records(records)
    if not rows:
        raise ValueError("montage records cannot be empty")
    if panel_width < 32:
        raise ValueError("panel_width must be at least 32 pixels")
    _require_fields(rows, (image_field, real_field, rendered_field))
    prepared: list[tuple[str, Image.Image, Image.Image]] = []
    max_height = 1
    for row in rows:
        real, rendered = _preview_pair(
            _load_image(row[real_field]),
            _load_image(row[rendered_field]),
        )
        resized = []
        for image in (real, rendered):
            height = max(1, round(image.height * panel_width / image.width))
            resized.append(
                image.resize((panel_width, height), Image.Resampling.LANCZOS)
            )
            max_height = max(max_height, height)
        prepared.append((str(row[image_field]), resized[0], resized[1]))

    margin = 12
    header_height = 34
    row_label_width = 100
    row_height = max_height + margin
    width = row_label_width + 2 * panel_width + 3 * margin
    height = header_height + len(prepared) * row_height + margin
    montage = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(montage)
    font = ImageFont.load_default()
    draw.text(
        (row_label_width + margin + panel_width // 2, margin),
        real_label,
        fill="black",
        font=font,
        anchor="ma",
    )
    draw.text(
        (row_label_width + 2 * margin + panel_width + panel_width // 2, margin),
        rendered_label,
        fill="black",
        font=font,
        anchor="ma",
    )
    for index, (image_id, real, rendered) in enumerate(prepared):
        y = header_height + index * row_height
        draw.text(
            (margin, y + max_height // 2),
            image_id,
            fill="black",
            font=font,
            anchor="lm",
        )
        for column, image in enumerate((real, rendered)):
            x = row_label_width + margin * (column + 1) + panel_width * column
            image_y = y + (max_height - image.height) // 2
            montage.paste(image, (x, image_y))
        if index + 1 < len(prepared):
            separator_y = y + row_height - margin // 2
            draw.line(
                (margin, separator_y, width - margin, separator_y),
                fill=(204, 204, 204),
            )

    png_destination = Path(png_path)
    png_destination.parent.mkdir(parents=True, exist_ok=True)
    montage.save(png_destination, format="PNG", compress_level=9, optimize=False)
    eps_destination = None
    if eps_path is not None:
        eps_destination = Path(eps_path)
        eps_destination.parent.mkdir(parents=True, exist_ok=True)
        montage.save(eps_destination, format="EPS")
    return png_destination, eps_destination
