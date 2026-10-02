"""Standard chart time windows, for multi_start(starts=...) and plot_from=."""

QUARTERS_PER_YEAR = 4
MONTHS_PER_YEAR = 12
RECENT_YEARS = 5
RECENT_MONTHS = 18  # a year and a half: a year ago is easy to find; 25 labelled bars was too cramped

# full history, then the recent window (plus the period growth starts from)
quarterly_plot_times = 0, -(QUARTERS_PER_YEAR * RECENT_YEARS) - 1
monthly_plot_times = 0, -RECENT_MONTHS - 1
