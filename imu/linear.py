"""Small incremental least-squares fit used by the IMU calibrations."""

import numpy as np


class LinearFit:
    """Fit ``output = coefficients @ features`` one sample at a time.

    Samples are accumulated until the feature matrix is observable.  The first
    estimate is an ordinary least-squares solution.  Later estimates use the
    Sherman--Morrison update, which is equivalent to refitting every sample.

    ``feature_scale`` is the expected magnitude of each feature.  It is used
    only to report a meaningful condition number; it does not change the fit.
    """

    def __init__(
        self,
        feature_count,
        output_count,
        feature_scale=None,
        max_condition_number=np.inf,
    ):
        if feature_count < 1 or output_count < 1:
            raise ValueError("feature and output counts must be positive")

        self.feature_count = int(feature_count)
        self.output_count = int(output_count)
        if feature_scale is None:
            feature_scale = np.ones(self.feature_count)
        self.feature_scale = np.asarray(feature_scale, dtype=float)
        if self.feature_scale.shape != (self.feature_count,):
            raise ValueError("feature_scale has the wrong length")
        if not np.isfinite(self.feature_scale).all() or np.any(
            self.feature_scale <= 0.0
        ):
            raise ValueError("feature_scale must be finite and positive")
        if max_condition_number < 1.0:
            raise ValueError("max_condition_number must be at least one")
        self.max_condition_number = float(max_condition_number)

        self.G = np.zeros((self.feature_count, self.feature_count))
        self.C = np.zeros((self.output_count, self.feature_count))
        self.sum_y2 = np.zeros(self.output_count)
        self.coefficients = None
        self._covariance = None
        self.count = 0

    @property
    def initialized(self):
        return self.coefficients is not None

    @property
    def covariance(self):
        """Return ``P = inverse(sum(x x^T))``, or ``None`` before fitting."""
        return self._covariance

    @property
    def condition_number(self):
        """Condition number of the scaled design matrix."""
        if np.linalg.matrix_rank(self.G) < self.feature_count:
            return np.inf
        scaled_gram = self.G / np.outer(
            self.feature_scale, self.feature_scale
        )
        # The Gram-matrix condition is the square of the design condition.
        return float(np.sqrt(np.linalg.cond(scaled_gram)))

    @property
    def indicators(self):
        """Return the article's raw ``sqrt(diag(P))`` indicators."""
        if self._covariance is None:
            return np.full(self.feature_count, np.inf)
        return np.sqrt(np.maximum(np.diag(self._covariance), 0.0))

    @property
    def residual_stddev(self):
        """Estimated standard deviation of each output residual."""
        degrees_of_freedom = self.count - self.feature_count
        if not self.initialized or degrees_of_freedom <= 0:
            return np.full(self.output_count, np.inf)

        residual_sum = self.sum_y2 - np.sum(
            self.coefficients * self.C,
            axis=1,
        )
        variance = residual_sum / degrees_of_freedom
        return np.sqrt(np.maximum(variance, 0.0))

    def update(self, features, output):
        features = np.asarray(features, dtype=float)
        output = np.atleast_1d(np.asarray(output, dtype=float))
        if features.shape != (self.feature_count,):
            raise ValueError("features have the wrong shape")
        if output.shape != (self.output_count,):
            raise ValueError("output has the wrong shape")
        if not np.isfinite(features).all() or not np.isfinite(output).all():
            raise ValueError("sample values must be finite")

        self.G += np.outer(features, features)
        self.C += np.outer(output, features)
        self.sum_y2 += np.square(output)
        self.count += 1

        if not self.initialized:
            condition = self.condition_number
            if (
                np.isfinite(condition)
                and condition <= self.max_condition_number
            ):
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
