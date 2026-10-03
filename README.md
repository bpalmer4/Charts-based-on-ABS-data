# au-econ: Australian Economic Data Charts

Charts of key Australian economic and social statistics, drawn from the latest data
published by the ABS, the RBA and other sources.

## Running it

The charts come from a Python package, `src/au_econ/`, run from the command line.
Start with `--list` to see what is available, for example:

```bash
uv run run.py --list            # every module, with its release names and topics
uv run run.py cpi               # one release, by short name or catalogue number (6401)
uv run run.py rba               # a topic: every module in it
uv run run.py somp --list       # the charts in a module
uv run run.py rba-fx --charts long_run_exchange_rates   # selected charts only
uv run run.py --topics          # the topic words and what they mean
uv run run.py --all             # everything
```

Each run names its own chart folder: `CHARTS/<release> - <title>/`, or
`CHARTS/<topic>/<release> - <title>/` for a topic. A full run first clears the images in
the folder it fills; a `--charts` run clears nothing.

Data come from the ABS, the RBA, the OECD, the BIS, FRED, the World Bank, DB.nomics,
Yahoo Finance, the EIA, OPEC and CME, other central banks and debt offices, and Australian
agencies (AIP, DCCEEW, ASIC, AFSA, Home Affairs). API keys live in `KEYS/` and downloads
are cached in `CACHE/` (both gitignored).

## The old world

The charts used to come from Jupyter notebooks in `notebooks/` (with helper modules beside
them, writing to `notebooks/CHARTS/`). They are frozen while the package is rebuilt
alongside them: each notebook is recreated in `src/au_econ/`, checked pixel-for-pixel
against the notebook's charts, then improved. When everything has been rebuilt, the
notebooks are deleted in one go. The design and the decisions behind it are in
[docs/restructure-spec.md](docs/restructure-spec.md); chart conventions are in its
section 11.

## Setup

The environment is managed with `uv`: `uv sync` installs the package (editable) and the
notebook tooling.
