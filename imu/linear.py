"""Incremental least-squares fit used by the IMU calibrations."""

import numpy as np


class LinearFit:
    """Fit ``output = coefficients @ features`` from a stream of samples."""

    def __init__(self, feature_scale=None, max_condition_number=np.inf):
        self.feature_scale = (
            None
            if feature_scale is None
            else np.asarray(feature_scale, dtype=float)
        )
        self.max_condition_number = float(max_condition_number)
        self.G = None
        self.C = None
        self.sum_y2 = None
        self.coefficients = None
        self._covariance = None
        self.count = 0

    @property
    def initialized(self):
        return self.coefficients is not None

    @property
    def covariance(self):
        return self._covariance

    @property
    def condition_number(self):
        if self.G is None or np.linalg.matrix_rank(self.G) < len(self.G):
            return np.inf
        scale = (
            np.ones(len(self.G))
            if self.feature_scale is None
            else self.feature_scale
        )
        scaled = self.G / np.outer(scale, scale)
        return float(np.sqrt(np.linalg.cond(scaled)))

    @property
    def indicators(self):
        if self._covariance is None:
            size = 0 if self.G is None else len(self.G)
            return np.full(size, np.inf)
        return np.sqrt(np.maximum(np.diag(self._covariance), 0.0))

    @property
    def residual_stddev(self):
        output_count = 0 if self.C is None else len(self.C)
        feature_count = 0 if self.G is None else len(self.G)
        degrees_of_freedom = self.count - feature_count
        if not self.initialized or degrees_of_freedom <= 0:
            return np.full(output_count, np.inf)
        residual_sum = self.sum_y2 - np.sum(
            self.coefficients * self.C,
            axis=1,
        )
        return np.sqrt(np.maximum(residual_sum / degrees_of_freedom, 0.0))

    def update(self, features, output):
        features = np.asarray(features, dtype=float).reshape(-1)
        output = np.asarray(output, dtype=float).reshape(-1)

        if self.G is None:
            self.G = np.zeros((features.size, features.size))
            self.C = np.zeros((output.size, features.size))
            self.sum_y2 = np.zeros(output.size)

        self.G += np.outer(features, features)
        self.C += np.outer(output, features)
        self.sum_y2 += output**2
        self.count += 1

        if not self.initialized:
            condition = self.condition_number
            if np.isfinite(condition) and condition <= self.max_condition_number:
                self._covariance = np.linalg.inv(self.G)
                self.coefficients = self.C @ self._covariance
            return self.coefficients

        projected = self._covariance @ features
        gain = projected / (1.0 + features @ projected)
        error = output - self.coefficients @ features
        self.coefficients += np.outer(error, gain)
        self._covariance -= np.outer(gain, projected)
        self._covariance = (self._covariance + self._covariance.T) / 2.0
        return self.coefficients
