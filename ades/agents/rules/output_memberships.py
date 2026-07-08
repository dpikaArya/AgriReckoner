"""
Output membership functions for fuzzy inference adjustments.
Each output variable represents an adjustment factor in [0, 1].
"""

OUTPUT_MEMBERSHIPS = {
    "Nitrogen_Adjustment": {
        "low": ("shouldered_z", 0.0, 0.35),
        "medium": ("triangle", 0.2, 0.5, 0.8),
        "high": ("shouldered_s", 0.6, 1.0),
    },
    "Potash_Adjustment": {
        "low": ("shouldered_z", 0.0, 0.35),
        "medium": ("triangle", 0.2, 0.5, 0.8),
        "high": ("shouldered_s", 0.6, 1.0),
    },
    "Irrigation_Adjustment": {
        "low": ("shouldered_z", 0.0, 0.35),
        "medium": ("triangle", 0.2, 0.5, 0.8),
        "high": ("shouldered_s", 0.6, 1.0),
    },
    "Dose_Adjustment": {
        "low": ("shouldered_z", 0.0, 0.35),
        "medium": ("triangle", 0.2, 0.5, 0.8),
        "high": ("shouldered_s", 0.6, 1.0),
    },
}
