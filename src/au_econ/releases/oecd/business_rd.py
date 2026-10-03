"""Business enterprise R&D (BERD) as a share of GDP: Australia against the OECD aggregate."""

# --- dependencies
import pandas as pd
from mgplot import line_plot_finalise

from au_econ.sources import dbnomics

# --- module contract
RELEASE = ("oecd-berd",)
TOPICS = ("international",)
TITLE = "Business R&D"

# --- constants
SERIES = {  # label: DBnomics path to the OECD Main Science and Technology Indicators series
    "OECD average": "OECD/DSD_MSTI@DF_MSTI/OECD.A.B.PT_B1GQ._Z._Z",
    "Australia": "OECD/DSD_MSTI@DF_MSTI/AUS.A.B.PT_B1GQ._Z._Z",
}


# --- data
def fetch() -> pd.DataFrame:
    """Return annual BERD as per cent of GDP, one column per SERIES label."""
    series = [dbnomics.get_series(path).rename(label) for label, path in SERIES.items()]
    return pd.concat(series, axis=1).sort_index()


# --- charts
def business_rd(data: pd.DataFrame) -> None:
    """BERD as a share of GDP: Australia against the OECD aggregate."""
    line_plot_finalise(
        data,
        annotate=True,
        rounding=2,
        title="Business R&D expenditure as a share of GDP: Australia vs OECD",
        ylabel="Per cent of GDP",
        rfooter="OECD: MSTI (via DBnomics)",
        lfooter="Business Enterprise R&D (BERD) as % of GDP. Australia: biennial. OECD aggregate: annual.",
        legend={"loc": "upper left", "fontsize": "small"},
        file_type="png",
    )


# --- table of contents, in run order
CHARTS = ((business_rd, ()),)
