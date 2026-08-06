"""Magnetometer simulator and linear calibration."""

import numpy as np

from .linear import LinearFit


def _vector(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or np.isnan(value).any():
        raise ValueError(f"{name} must be a three-element vector")
    return value


class MagnetometerParameters:
    """Parameters of ``measured = matrix @ reference + offset + noise``."""

    def __init__(
        self,
        transformation_matrix=None,
        offset=None,
        noise_stddev=None,
    ):
        if transformation_matrix is None:
            transformation_matrix = np.eye(3)
        if offset is None:
            offset = np.zeros(3)
        if noise_stddev is None:
            noise_stddev = np.zeros(3)

        self.transformation_matrix = np.asarray(
            transformation_matrix,
            dtype=float,
        )
        if (
            self.transformation_matrix.shape != (3, 3)
            or not np.isfinite(self.transformation_matrix).all()
            or np.linalg.matrix_rank(self.transformation_matrix) < 3
        ):
            raise ValueError(
                "transformation_matrix must be a nonsingular 3x3 matrix"
            )
        self.offset = _vector(offset, "offset")
        self.noise_stddev = _vector(noise_stddev, "noise_stddev")
        if np.any(self.noise_stddev < 0.0):
            raise ValueError("noise_stddev must be non-negative")

    def apply(self, reference):
        return self.transformation_matrix @ np.asarray(reference) + self.offset

    def correct(self, reading):
        return np.linalg.solve(
            self.transformation_matrix,
            np.asarray(reading) - self.offset,
        )


class Magnetometer:
    def __init__(self, parameters=None, seed=42):
        self.parameters = parameters or MagnetometerParameters()
        if not np.isfinite(self.parameters.noise_stddev).all():
            raise ValueError("simulator noise_stddev must be finite")
        self._rng = np.random.default_rng(seed)

    def measure(self, reference_body):
        reference_body = _vector(reference_body, "reference_body")
        noise = self._rng.normal(0.0, self.parameters.noise_stddev)
        return self.parameters.apply(reference_body) + noise


class MagnetometerCalibration:
    """Fit either the complete model or the article's near-level model."""

    def __init__(
        self,
        constrained=False,
        indicator_limit=np.inf,
        offset_indicator_limit=np.inf,
        max_condition_number=1e10,
        consecutive_updates=1,
        reference_scale_ut=50.0,
    ):
        self.constrained = constrained
        self.indicator_limit = float(indicator_limit)
        self.offset_indicator_limit = float(offset_indicator_limit)
        self.max_condition_number = float(max_condition_number)
        self.consecutive_updates = int(consecutive_updates)
        self.stable_updates = 0
        self.converged = False

        if constrained:
            self.horizontal_fit = LinearFit(
                3,
                2,
                [reference_scale_ut, reference_scale_ut, 1.0],
                max_condition_number,
            )
            self.vertical_fit = LinearFit(
                3,
                1,
                [reference_scale_ut] * 3,
                max_condition_number,
            )
        else:
            self.fit = LinearFit(
                4,
                3,
                [reference_scale_ut] * 3 + [1.0],
                max_condition_number,
            )

    @property
    def initialized(self):
        if self.constrained:
            return (
                self.horizontal_fit.initialized
                and self.vertical_fit.initialized
            )
        return self.fit.initialized

    @property
    def condition_number(self):
        if self.constrained:
            return max(
                self.horizontal_fit.condition_number,
                self.vertical_fit.condition_number,
            )
        return self.fit.condition_number

    def update(self, reference_body, reading):
        reference = _vector(reference_body, "reference_body")
        reading = _vector(reading, "reading")
        if self.constrained:
            self.horizontal_fit.update(
                [reference[0], reference[1], 1.0],
                reading[:2],
            )
            self.vertical_fit.update(reference, reading[2])
        else:
            self.fit.update(np.append(reference, 1.0), reading)
        self._update_stopping_state()
        return self.get_parameters() if self.initialized else None

    def _update_stopping_state(self):
        if self.converged:
            return
        if self.constrained:
            stable = (
                self.initialized
                and self.condition_number <= self.max_condition_number
                and np.all(
                    self.horizontal_fit.indicators[:2]
                    <= self.indicator_limit
                )
                and self.horizontal_fit.indicators[2]
                <= self.offset_indicator_limit
                and np.all(
                    self.vertical_fit.indicators <= self.indicator_limit
                )
            )
        else:
            stable = (
                self.initialized
                and self.condition_number <= self.max_condition_number
                and np.all(self.fit.indicators[:3] <= self.indicator_limit)
                and self.fit.indicators[3] <= self.offset_indicator_limit
            )
        self.stable_updates = self.stable_updates + 1 if stable else 0
        self.converged = self.stable_updates >= self.consecutive_updates

    def get_parameters(self):
        if not self.initialized:
            raise RuntimeError("calibration is not initialized")
        if self.constrained:
            matrix = np.zeros((3, 3))
            offset = np.zeros(3)
            matrix[:2, :2] = self.horizontal_fit.coefficients[:, :2]
            offset[:2] = self.horizontal_fit.coefficients[:, 2]
            matrix[2] = self.vertical_fit.coefficients[0]
            noise = np.append(
                self.horizontal_fit.residual_stddev,
                self.vertical_fit.residual_stddev,
            )
        else:
            matrix = self.fit.coefficients[:, :3]
            offset = self.fit.coefficients[:, 3]
            noise = self.fit.residual_stddev
        return MagnetometerParameters(matrix, offset, noise)

    def correct(self, reading):
        return self.get_parameters().correct(reading)
