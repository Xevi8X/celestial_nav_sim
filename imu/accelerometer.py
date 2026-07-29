from dataclasses import dataclass, field

import numpy as np

from common.config import Config


@dataclass
class AccelerometerParameters:
    assembly_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    axis_scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    noise_stddev: np.ndarray = field(default_factory=lambda: np.zeros(3))


class Accelerometer:
    def __init__(self, parameters: AccelerometerParameters, seed: int = 42):
        self._p = parameters
        self._noise_rng = np.random.default_rng(seed)

    def measure(
        self, R_bw: np.ndarray, acceleration_ned: np.ndarray
    ) -> np.ndarray:
        gravity_ned = np.array([0.0, 0.0, Config.GRAVITY])
        acceleration_body = (
            acceleration_ned - gravity_ned
        ) @ R_bw.T
        noise = self._noise_rng.normal(0.0, self._p.noise_stddev)
        return (
            (acceleration_body + self._p.bias) * self._p.axis_scale
        ) @ self._p.assembly_matrix + noise


class AccelerometerCalibration:
    def __init__(self):
        self._theta = np.vstack((np.eye(3), np.zeros(3)))
        self._gram = np.zeros((4, 4))
        self._reference_mean = np.zeros(3)
        self._reading_mean = np.zeros(3)
        self._reference_scatter = np.zeros((3, 3))
        self._cross_scatter = np.zeros((3, 3))
        self._reading_scatter = np.zeros((3, 3))
        self._noise_covariance = None
        self._samples = 0

    def _ref(
        self,
        R_bw: np.ndarray,
        acceleration_ned: np.ndarray = None,
    ) -> np.ndarray:
        if acceleration_ned is None:
            acceleration_ned = np.zeros(3)
        gravity_ned = np.array([0.0, 0.0, Config.GRAVITY])
        return (acceleration_ned - gravity_ned) @ R_bw.T

    def update(
        self,
        R_bw: np.ndarray,
        reading: np.ndarray,
        acceleration_ned: np.ndarray = None,
    ) -> AccelerometerParameters:
        reference = self._ref(R_bw, acceleration_ned)
        x = np.append(reference, 1.0)
        self._gram += np.outer(x, x)
        self._samples += 1

        reference_delta = reference - self._reference_mean
        reading_delta = reading - self._reading_mean
        self._reference_mean += reference_delta / self._samples
        self._reading_mean += reading_delta / self._samples
        self._reference_scatter += np.outer(
            reference_delta,
            reference - self._reference_mean,
        )
        self._cross_scatter += np.outer(
            reference_delta,
            reading - self._reading_mean,
        )
        self._reading_scatter += np.outer(
            reading_delta,
            reading - self._reading_mean,
        )

        if np.linalg.matrix_rank(self._reference_scatter) == 3:
            R = np.linalg.solve(
                self._reference_scatter,
                self._cross_scatter,
            )
            offset = self._reading_mean - self._reference_mean @ R
            self._theta = np.vstack((R, offset))
            if self._samples > 4:
                residual = (
                    self._reading_scatter
                    - self._cross_scatter.T @ R
                )
                covariance = residual / (self._samples - 4)
                covariance = (covariance + covariance.T) / 2.0
                values, vectors = np.linalg.eigh(covariance)
                self._noise_covariance = (
                    vectors * np.maximum(values, 0.0)
                ) @ vectors.T
        return self.get_parameters()

    def get_condition_number(self) -> float:
        if self._samples < 4 or np.linalg.matrix_rank(self._gram) < 4:
            return np.inf
        normalizer = np.diag([
            1.0 / Config.GRAVITY,
            1.0 / Config.GRAVITY,
            1.0 / Config.GRAVITY,
            1.0,
        ])
        normalized_gram = normalizer @ self._gram @ normalizer
        return np.sqrt(np.linalg.cond(normalized_gram))

    def get_variance(self) -> np.ndarray:
        if self._noise_covariance is None:
            return np.full((4, 3), np.inf)
        gram_inverse = np.linalg.inv(self._gram)
        matrix_variance = (
            np.diag(gram_inverse)[:3, None]
            * np.diag(self._noise_covariance)[None, :]
        )
        R = self._theta[:3]
        bias = np.linalg.solve(R.T, self._theta[3])
        h = np.append(-bias, 1.0)
        offset_covariance = (
            h @ gram_inverse @ h
        ) * self._noise_covariance
        R_inverse = np.linalg.inv(R)
        bias_covariance = (
            R_inverse.T @ offset_covariance @ R_inverse
        )
        return np.vstack(
            (
                matrix_variance,
                np.diag(bias_covariance),
            )
        )

    def get_parameters(self) -> AccelerometerParameters:
        R = self._theta[:3]
        offset = self._theta[3]
        axis_scale = np.linalg.norm(R, axis=1)
        assembly_matrix = R / axis_scale[:, None]
        bias = np.linalg.solve(R.T, offset)
        noise_variance = (
            np.diag(self._noise_covariance)
            if self._noise_covariance is not None
            else np.full(3, np.inf)
        )
        return AccelerometerParameters(
            assembly_matrix=assembly_matrix,
            axis_scale=axis_scale,
            bias=bias,
            noise_stddev=np.sqrt(noise_variance),
        )