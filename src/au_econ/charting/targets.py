"""RBA inflation target markers for charts (pass to finalise as axhspan= / axhline=)."""

ANNUAL_TARGET_PCT = 2.5
QUARTERS_PER_YEAR = 4
MONTHS_PER_YEAR = 12

ANNUAL_CPI_TARGET_RANGE: dict[str, float | str | int] = {
    "ymin": 2,
    "ymax": 3,
    "color": "#dddddd",
    "label": "2-3% annual inflation target range",
    "zorder": -1,
}

QUARTERLY_CPI_TARGET: dict[str, float | str | int] = {
    "y": (pow(1 + ANNUAL_TARGET_PCT / 100, 1 / QUARTERS_PER_YEAR) - 1) * 100,
    "linestyle": "dashed",
    "linewidth": 0.75,
    "color": "darkred",
    "label": "Quarterly growth consistent with 2.5% annual inflation",
}

MONTHLY_CPI_TARGET: dict[str, float | str | int] = {
    "y": (pow(1 + ANNUAL_TARGET_PCT / 100, 1 / MONTHS_PER_YEAR) - 1) * 100,
    "color": "darkred",
    "linewidth": 0.75,
    "linestyle": "--",
    "label": "Monthly growth consistent with a 2.5% annual inflation target",
    "zorder": -1,
}
