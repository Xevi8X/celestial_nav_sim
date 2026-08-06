# Article experiments

Run the complete validation chain from the repository root with:

```sh
.venv/bin/python bin/article_validation.py
```

This entry point runs the four experiments sequentially. The renderer's
measured angular residual is passed to both calibration simulations, and the
estimated accelerometer calibration is passed to static-position estimation.

Each experiment also remains independently runnable with fixed defaults:

```sh
.venv/bin/python -m experiments.renderer_tuning
.venv/bin/python -m experiments.accelerometer_calibration
.venv/bin/python -m experiments.magnetometer_calibration
.venv/bin/python -m experiments.static_position
```

Their outputs are kept separately:

| Experiment | Results folder |
| --- | --- |
| Renderer tuning | `results/renderer_tuning/` |
| Accelerometer calibration | `results/accelerometer_calibration/` |
| Magnetometer calibration | `results/magnetometer_calibration/` |
| Static-position estimation | `results/static_position/` |

Tables and raw observations are written as CSV files. Plots use Matplotlib's
default theme and are saved as paired EPS and PNG files.

Renderer tuning fits `.data/image1.fit`, `.data/image2.fit`, and
`.data/image3.fit` independently. Their camera parameters are averaged, and
the average model renders all three real-image comparisons.

The simulations use Warsaw, New York, Sao Paulo, and Sydney. Independent
calibration runs use a default `0.0022` degree attitude-noise standard
deviation. In the complete chain this value is recalculated from the three
photos. Static-position estimation uses the matching converged accelerometer
estimate from the `0` degree initial-azimuth calibration; its independent run
uses the imposed calibrated model.
