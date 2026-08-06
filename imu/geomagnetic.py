"""World Magnetic Model references used by the magnetometer experiment."""

from dataclasses import dataclass
import datetime

import numpy as np


@dataclass
class GeomagneticField:
    north_nT: float
    east_nT: float
    down_nT: float
    decimal_year: float = np.nan
    model: str = ""
    in_blackout_zone: bool = False
    in_caution_zone: bool = False

    @property
    def ned_nT(self):
        return np.array([self.north_nT, self.east_nT, self.down_nT])

    @property
    def ned_uT(self):
        return self.ned_nT / 1000.0

    @property
    def horizontal_intensity_nT(self):
        return float(np.hypot(self.north_nT, self.east_nT))

    @property
    def total_intensity_nT(self):
        return float(np.linalg.norm(self.ned_nT))

    @property
    def declination_deg(self):
        return float(np.degrees(np.arctan2(self.east_nT, self.north_nT)))

    @property
    def inclination_deg(self):
        return float(np.degrees(np.arctan2(
            self.down_nT,
            self.horizontal_intensity_nT,
        )))

    @classmethod
    def from_elements(cls, total_intensity_nT, declination_deg, inclination_deg):
        declination = np.radians(declination_deg)
        inclination = np.radians(inclination_deg)
        horizontal = total_intensity_nT * np.cos(inclination)
        return cls(
            horizontal * np.cos(declination),
            horizontal * np.sin(declination),
            total_intensity_nT * np.sin(inclination),
        )


def _decimal_year(value):
    if isinstance(value, datetime.datetime):
        start = datetime.datetime(value.year, 1, 1, tzinfo=value.tzinfo)
        end = datetime.datetime(value.year + 1, 1, 1, tzinfo=value.tzinfo)
        elapsed = (value - start).total_seconds()
        duration = (end - start).total_seconds()
        return value.year + elapsed / duration
    if isinstance(value, datetime.date):
        start = datetime.date(value.year, 1, 1)
        end = datetime.date(value.year + 1, 1, 1)
        return value.year + (value - start).days / (end - start).days
    return float(value)


class WMM2025Provider:
    def __init__(self, model=None):
        if model is None:
            try:
                from pygeomag import GeoMag
                from pygeomag.wmm.wmm_2025 import WMM_2025
            except ImportError as error:
                raise ImportError(
                    "WMM2025Provider requires pygeomag>=1.1"
                ) from error
            model = GeoMag(coefficients_data=WMM_2025)
        self.model = model

    def field(self, latitude_deg, longitude_deg, altitude_m, time):
        year = _decimal_year(time)
        result = self.model.calculate(
            glat=float(latitude_deg),
            glon=float(longitude_deg),
            alt=float(altitude_m) / 1000.0,
            time=year,
            allow_date_outside_lifespan=False,
            raise_in_warning_zone=False,
        )
        return GeomagneticField(
            result.x,
            result.y,
            result.z,
            decimal_year=year,
            model=str(getattr(self.model, "model", "WMM-2025")),
            in_blackout_zone=bool(
                getattr(result, "in_blackout_zone", False)
            ),
            in_caution_zone=bool(
                getattr(result, "in_caution_zone", False)
            ),
        )


def magnetic_reference_body(field_ned, navigation_to_body):
    if isinstance(field_ned, GeomagneticField):
        field_ned = field_ned.ned_nT
    return np.asarray(navigation_to_body) @ np.asarray(field_ned)


def magnetic_heading_deg(field, minimum_horizontal=1e-12):
    """Return clockwise heading from north; the input unit is irrelevant."""
    field = np.asarray(field, dtype=float)
    if np.hypot(field[0], field[1]) <= minimum_horizontal:
        raise ValueError("magnetic heading is undefined for this field vector")
    return float(np.degrees(np.arctan2(field[1], field[0])) % 360.0)
