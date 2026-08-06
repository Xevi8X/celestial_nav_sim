"""Accelerometer simulator and linear calibration."""

import numpy as np

from common.config import Config

from .linear import LinearFit


def _vector(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or np.isnan(value).any():
        raise ValueError(f"{name} must be a three-element vector")
    return value


def _matrix(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    return value


class AccelerometerParameters:
    """Parameters of ``measured = R @ S @ (reference + bias) + noise``."""

    def __init__(
        self,
        assembly_matrix=None,
        axis_scale=None,
        bias=None,
        noise_stddev=None,
    ):
        if assembly_matrix is None:
            assembly_matrix = np.eye(3)
        if axis_scale is None:
            axis_scale = np.ones(3)
        if bias is None:
            bias = np.zeros(3)
        if noise_stddev is None:
            noise_stddev = np.zeros(3)

        self.assembly_matrix = _matrix(assembly_matrix, "assembly_matrix")
        if np.linalg.matrix_rank(self.assembly_matrix) < 3:
            raise ValueError("assembly_matrix must be nonsingular")
        self.axis_scale = _vector(axis_scale, "axis_scale")
        self.bias = _vector(bias, "bias")
        self.noise_stddev = _vector(noise_stddev, "noise_stddev")
        if np.any(self.axis_scale <= 0.0):
            raise ValueError("axis_scale must be positive")
        if np.any(self.noise_stddev < 0.0):
            raise ValueError("noise_stddev must be non-negative")

    @property
    def linear_matrix(self):
        return self.assembly_matrix @ np.diag(self.axis_scale)

    def apply(self, reference):
        return self.linear_matrix @ (np.asarray(reference) + self.bias)

    def correct(self, reading):
        offset = self.linear_matrix @ self.bias
        return np.linalg.solve(self.linear_matrix, np.asarray(reading) - offset)


def _specific_force(navigation_to_body, acceleration_ned=None):
    navigation_to_body = _matrix(navigation_to_body, "navigation_to_body")
    if acceleration_ned is None:
        acceleration_ned = np.zeros(3)
    acceleration_ned = _vector(acceleration_ned, "acceleration_ned")
    gravity_ned = np.array([0.0, 0.0, Config.GRAVITY])
    return navigation_to_body @ (acceleration_ned - gravity_ned)


class Accelerometer:
    def __init__(self, parameters=None, seed=42):
        self.parameters = parameters or AccelerometerParameters()
        if not np.isfinite(self.parameters.noise_stddev).all():
            raise ValueError("simulator noise_stddev must be finite")
        self._rng = np.random.default_rng(seed)

    def measure(self, navigation_to_body, acceleration_ned=None):
        reference = _specific_force(navigation_to_body, acceleration_ned)
        noise = self._rng.normal(0.0, self.parameters.noise_stddev)
        return self.parameters.apply(reference) + noise


class AccelerometerCalibration:
    """Fit the article's four affine accelerometer coefficients online."""

    def __init__(
        self,
        matrix_indicator_limit=np.inf,
        bias_indicator_limit=np.inf,
        max_condition_number=1e10,
        consecutive_updates=1,
    ):
        self.fit = LinearFit(
            4,
            3,
            feature_scale=[Config.GRAVITY] * 3 + [1.0],
            max_condition_number=max_condition_number,
        )
        self.max_condition_number = float(max_condition_number)
        self.matrix_indicator_limit = float(matrix_indicator_limit)
        self.bias_indicator_limit = float(bias_indicator_limit)
        self.consecutive_updates = int(consecutive_updates)
        self.stable_updates = 0
        self.converged = False

    @property
    def initialized(self):
        return self.fit.initialized

    @property
    def condition_number(self):
        return self.fit.condition_number

    @property
    def indicators(self):
        return self.fit.indicators

    def update(self, navigation_to_body, reading, acceleration_ned=None):
        reference = _specific_force(navigation_to_body, acceleration_ned)
        return self.update_reference(reference, reading)

    def update_reference(self, reference, reading):
        features = np.append(_vector(reference, "reference"), 1.0)
        self.fit.update(features, _vector(reading, "reading"))
        self._update_stopping_state()
        return self.get_parameters() if self.initialized else None

    def _update_stopping_state(self):
        if self.converged:
            return
        stable = (
            self.initialized
            and self.condition_number <= self.max_condition_number
            and np.all(self.indicators[:3] <= self.matrix_indicator_limit)
            and self.indicators[3] <= self.bias_indicator_limit
        )
        self.stable_updates = self.stable_updates + 1 if stable else 0
        self.converged = self.stable_updates >= self.consecutive_updates

    def get_condition_number(self):
        return self.condition_number

    def get_variance(self):
        """Coefficient variance, with input-bias variance in the last row."""
        if not self.initialized:
            return np.full((4, 3), np.inf)
        noise_variance = np.square(self.fit.residual_stddev)
        if not np.isfinite(noise_variance).all():
            return np.full((4, 3), np.inf)

        coefficient_variance = (
            np.diag(self.fit.covariance)[:, None] * noise_variance[None, :]
        )
        matrix = self.fit.coefficients[:, :3]
        bias = np.linalg.solve(matrix, self.fit.coefficients[:, 3])
        bias_regressor = np.append(-bias, 1.0)
        output_variance = (
            bias_regressor @ self.fit.covariance @ bias_regressor
        ) * noise_variance
        inverse = np.linalg.inv(matrix)
        bias_covariance = inverse @ np.diag(output_variance) @ inverse.T
        coefficient_variance[3] = np.diag(bias_covariance)
        return coefficient_variance

    def get_parameters(self):
        if not self.initialized:
            raise RuntimeError("calibration is not initialized")
        matrix = self.fit.coefficients[:, :3]
        offset = self.fit.coefficients[:, 3]
        scale = np.linalg.norm(matrix, axis=0)
        normalized = matrix / scale
        left, _, right = np.linalg.svd(normalized)
        correction = np.eye(3)
        correction[2, 2] = np.sign(np.linalg.det(left @ right))
        rotation = left @ correction @ right
        return AccelerometerParameters(
            assembly_matrix=rotation,
            axis_scale=scale,
            bias=np.linalg.solve(matrix, offset),
            noise_stddev=self.fit.residual_stddev,
        )

    def correct(self, reading):
        if not self.initialized:
            raise RuntimeError("calibration is not initialized")
        matrix = self.fit.coefficients[:, :3]
        offset = self.fit.coefficients[:, 3]
        return np.linalg.solve(matrix, np.asarray(reading) - offset)
