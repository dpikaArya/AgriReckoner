"""
Fuzzy membership functions for agricultural variables.
Supports triangular, trapezoidal, gaussian, and shouldered shapes.
"""

import numpy as np


def triangle(x: float | np.ndarray, a: float, b: float, c: float) -> float | np.ndarray:
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-10), (c - x) / (c - b + 1e-10)))


def trapezoid(x: float | np.ndarray, a: float, b: float, c: float, d: float) -> float | np.ndarray:
    return np.maximum(
        0, np.minimum(np.minimum((x - a) / (b - a + 1e-10), 1), (d - x) / (d - c + 1e-10))
    )


def gaussian(x: float | np.ndarray, mean: float, sigma: float) -> float | np.ndarray:
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def shouldered_s(x: float | np.ndarray, a: float, b: float) -> float | np.ndarray:
    """S-shaped (high) shoulder: 0 at a, rising to 1 at b, then flat at 1."""
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    result = np.where(x <= a, 0, result)
    result = np.where(x >= b, 1, result)
    ramp = (x - a) / (b - a + 1e-10)
    result = np.where((x > a) & (x < b), ramp, result)
    return result


def shouldered_z(x: float | np.ndarray, a: float, b: float) -> float | np.ndarray:
    """Z-shaped (low) shoulder: 1 at x <= a, falling to 0 at x >= b."""
    return 1 - shouldered_s(x, a, b)


def clamp_membership(v: float) -> float:
    return float(np.clip(v, 0, 1))


ALL_MEMBERSHIP_FUNCTIONS = {}

VAR_MEMBERSHIPS = {
    "Nitrogen": {
        "low": ("shouldered_z", 30, 60),
        "medium": ("triangle", 30, 60, 100),
        "high": ("shouldered_s", 80, 120),
    },
    "Phosphorus": {
        "low": ("shouldered_z", 10, 20),
        "medium": ("triangle", 10, 25, 40),
        "high": ("shouldered_s", 30, 50),
    },
    "Potassium": {
        "low": ("shouldered_z", 80, 120),
        "medium": ("triangle", 80, 150, 250),
        "high": ("shouldered_s", 180, 300),
    },
    "Soil_pH": {
        "acidic": ("shouldered_z", 5.5, 6.0),
        "neutral": ("trapezoid", 5.5, 6.5, 7.5, 8.0),
        "alkaline": ("shouldered_s", 7.5, 8.0),
    },
    "Rainfall": {
        "low": ("shouldered_z", 400, 600),
        "medium": ("triangle", 400, 750, 1100),
        "high": ("shouldered_s", 900, 1200),
    },
    "Temperature_Max": {
        "cool": ("shouldered_z", 20, 25),
        "moderate": ("trapezoid", 20, 25, 32, 35),
        "hot": ("shouldered_s", 32, 38),
    },
    "Organic_Carbon": {
        "low": ("shouldered_z", 0.4, 0.6),
        "medium": ("triangle", 0.4, 0.75, 1.0),
        "high": ("shouldered_s", 0.8, 1.2),
    },
    "Growth_Stage": {
        "seedling": ("shouldered_z", 15, 30),
        "vegetative": ("trapezoid", 20, 35, 50, 65),
        "flowering": ("trapezoid", 50, 60, 75, 85),
        "maturity": ("shouldered_s", 75, 95),
    },
    "Zinc": {
        "low": ("shouldered_z", 0.5, 1.0),
        "medium": ("triangle", 0.5, 1.5, 2.5),
        "high": ("shouldered_s", 2.0, 3.0),
    },
    "Yield_Prediction": {
        "low": ("shouldered_z", 2.0, 3.5),
        "medium": ("triangle", 2.5, 4.5, 6.5),
        "high": ("shouldered_s", 5.5, 8.0),
    },
}


def fuzzify(variable: str, value: float) -> dict[str, float]:
    if variable not in VAR_MEMBERSHIPS:
        return {}
    result = {}
    for label, (shape, *params) in VAR_MEMBERSHIPS[variable].items():
        if shape == "triangle":
            a, b, c = params
            val = triangle(value, a, b, c)
        elif shape == "trapezoid":
            a, b, c, d = params
            val = trapezoid(value, a, b, c, d)
        elif shape == "gaussian":
            mean, sigma = params
            val = gaussian(value, mean, sigma)
        elif shape == "shouldered_s":
            a, b = params
            val = shouldered_s(value, a, b)
        elif shape == "shouldered_z":
            a, b = params
            val = shouldered_z(value, a, b)
        else:
            val = 0.0
        result[label] = clamp_membership(val)
    return result
