"""Model functions used by voltage and bowl calibration workflows."""

from __future__ import annotations

import numpy as np
from collections.abc import Mapping


def voltage_corr(x, a, b, c):
    """Quadratic voltage correction model."""
    return a + b * x + c * (x ** 2)


def bowl_corr(data_xy, a, b, c, d, e, f):
    """Quadratic bowl correction model."""
    x = data_xy[0]
    y = data_xy[1]
    return a + b * x + c * y + d * (x ** 2) + e * x * y + f * (y ** 2)


def bowl_corr_radial(data_xy, a, b, c, d, e):
    """Radial-dominant bowl correction model using r^2 as the primary term."""
    x = np.asarray(data_xy[0], dtype=float)
    y = np.asarray(data_xy[1], dtype=float)
    r2 = x ** 2 + y ** 2
    return a + b * r2 + c * (r2 ** 2) + d * x + e * y


def robust_voltage_fit(dld_high_voltage, dld_t):
    """Perform robust polynomial fitting using a RANSAC pipeline."""
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import RANSACRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    x_values = dld_high_voltage.reshape(-1, 1)
    y_values = dld_t
    polynomial = PolynomialFeatures(degree=2)
    feature_count = polynomial.fit_transform(np.zeros((1, x_values.shape[1]))).shape[1]
    if x_values.shape[0] <= feature_count:
        model = make_pipeline(polynomial, LinearRegression())
    else:
        model = make_pipeline(
            polynomial,
            RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=min(x_values.shape[0], feature_count),
                random_state=42,
            ),
        )
    model.fit(x_values, y_values)
    return model


def hybrid_calibration_model(dld_x, dld_y, dld_t):
    """Train a random-forest regression model for bowl correction."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    x_values = np.column_stack((dld_x, dld_y))
    y_values = 1 / dld_t
    x_train, x_test, y_train, y_test = train_test_split(
        x_values, y_values, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(x_train, y_train)

    score = model.score(x_test, y_test)
    print(f"Machine learning model R2 score: {score:.3f}")
    return model


def robust_fit(dld_x, dld_y, dld_t, degree=2):
    """Perform robust polynomial fitting for bowl correction."""
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import RANSACRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    x_values = np.column_stack((dld_x, dld_y))
    y_values = dld_t
    polynomial = PolynomialFeatures(degree=degree)
    feature_count = polynomial.fit_transform(np.zeros((1, x_values.shape[1]))).shape[1]
    if x_values.shape[0] <= feature_count:
        model = make_pipeline(polynomial, LinearRegression())
    else:
        model = make_pipeline(
            polynomial,
            RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=min(x_values.shape[0], feature_count),
                random_state=42,
            ),
        )
    model.fit(x_values, y_values)
    return model


def _predict_voltage_model(model_name, fitresult, voltage_values):
    """Predict voltage correction values for the selected model."""
    if model_name == "curve_fit":
        return voltage_corr(voltage_values, *fitresult)
    return fitresult.predict(voltage_values.reshape(-1, 1))


def _predict_bowl_model(fit_mode, parameters, dld_x, dld_y):
    """Predict bowl correction values for the selected fitting mode."""
    if isinstance(parameters, Mapping) and parameters.get("model") in {"radial_curve_fit", "radial_linear"}:
        coeffs = np.asarray(parameters.get("parameters", ()), dtype=float)
        if coeffs.shape[0] != 5:
            raise ValueError("Radial bowl model requires exactly 5 parameters")
        return bowl_corr_radial([dld_x, dld_y], *coeffs)

    if fit_mode == "curve_fit":
        return bowl_corr([dld_x, dld_y], *parameters)
    return parameters.predict(np.column_stack((dld_x, dld_y)))


__all__ = [
    "voltage_corr",
    "bowl_corr",
    "bowl_corr_radial",
    "robust_voltage_fit",
    "hybrid_calibration_model",
    "robust_fit",
    "_predict_voltage_model",
    "_predict_bowl_model",
]
