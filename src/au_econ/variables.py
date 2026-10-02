"""Shared short economic names for `run.py --charts`, with their meanings.

A chart answers to its function name and to any of these names listed beside it in its
module's CHARTS. Names are lowercase ASCII. One name may cover several measures (headline,
trimmed mean and weighted median charts can all answer to `pi`). A name is added here the
first time a chart uses it.
"""

VARIABLES: dict[str, str] = {}
