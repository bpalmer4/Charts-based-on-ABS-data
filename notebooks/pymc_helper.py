"""Common helper functions for PyMC Bayesian models."""

import math
from collections.abc import Sequence

import arviz as az
import matplotlib.pyplot as plt
import mgplot as mg
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy import stats


def check_model_diagnostics(trace: az.InferenceData) -> None:
    """Check the inference data for potential problems.

    Diagnostics applied:
    - R-hat (Gelman-Rubin): Compares between-chain and within-chain variance.
      Values > 1.01 suggest chains have not converged to the same distribution.
    - ESS (Effective Sample Size): Estimates independent samples accounting for
      autocorrelation. Low ESS (< 400) indicates high autocorrelation or short chains.
    - MCSE/sd ratio: Monte Carlo standard error relative to posterior sd.
      Ratios > 5% suggest insufficient samples for reliable posterior mean estimates.
    - Divergent transitions: Indicate regions where the sampler struggled with
      posterior geometry. Any divergences may signal biased estimates.
    - Tree depth saturation: High rates at max tree depth suggest the sampler
      is working harder than expected, possibly due to difficult geometry.
    - BFMI (Bayesian Fraction of Missing Information): Measures how well the
      sampler explores the energy distribution. Values < 0.3 suggest poor exploration.
    """

    def warn(w: bool) -> str:
        return "--- THERE BE DRAGONS ---> " if w else ""

    summary = az.summary(trace)

    # check model convergence
    max_r_hat = 1.01
    statistic = summary.r_hat.max()
    print(
        f"{warn(statistic > max_r_hat)}Maximum R-hat convergence diagnostic: {statistic}"
    )

    # check effective sample size
    min_ess = 400
    statistic = summary[["ess_tail", "ess_bulk"]].min().min()
    print(
        f"{warn(statistic < min_ess)}Minimum effective sample size (ESS) estimate: {int(statistic)}"
    )

    # check MCSE ratio (should be < 5% of posterior sd)
    max_mcse_ratio = 0.05
    statistic = (summary["mcse_mean"] / summary["sd"]).max()
    print(
        f"{warn(statistic > max_mcse_ratio)}Maximum MCSE/sd ratio: {statistic:0.3f}"
    )

    # check for divergences (rate-based: < 1 in 10,000 samples)
    # Note: even one divergence might be problematic.
    max_divergence_rate = 1 / 10_000  # 0.01%
    try:
        diverging_count = int(np.sum(trace.sample_stats.diverging))
    except (ValueError, AttributeError):
        diverging_count = 0
    total_samples = trace.posterior.sizes["draw"] * trace.posterior.sizes["chain"]
    divergence_rate = diverging_count / total_samples
    print(
        f"{warn(divergence_rate > max_divergence_rate)}Divergent transitions: "
        f"{diverging_count}/{total_samples} ({divergence_rate:.4%})"
    )

    # check max tree depth saturation
    max_tree_depth_rate = 0.05  # warn if > 5% at max depth
    try:
        # Use reached_max_treedepth if available (preferred - compares to configured max)
        if hasattr(trace.sample_stats, "reached_max_treedepth"):
            at_max = trace.sample_stats.reached_max_treedepth.values
            at_max_rate = float(at_max.mean())
            max_observed = int(trace.sample_stats.tree_depth.values.max())
            print(
                f"{warn(at_max_rate >= max_tree_depth_rate)}Tree depth at configured max: "
                f"{at_max_rate:.2%} (max observed: {max_observed})"
            )
        else:
            # Fallback: compare to observed max (less reliable)
            tree_depth = trace.sample_stats.tree_depth.values
            max_depth = int(tree_depth.max())
            at_max_rate = (tree_depth == max_depth).mean()
            print(
                f"{warn(at_max_rate >= max_tree_depth_rate)}Tree depth at max ({max_depth}): "
                f"{at_max_rate:.2%} (note: comparing to observed max, not configured)"
            )
    except AttributeError:
        pass  # Not all samplers report tree depth

    # check BFMI
    min_bfmi = 0.3
    statistic = az.bfmi(trace).min()
    print(
        f"{warn(statistic < min_bfmi)}Minimum Bayesian fraction of missing information: {statistic:0.2f}"
    )


def get_vector_var(var_name: str, trace: az.InferenceData) -> pd.DataFrame:
    """Extract chains/draws for a vector variable.

    Returns DataFrame with rows=time periods, columns=samples.
    """
    return (
        az.extract(trace, var_names=var_name)
        .transpose("sample", ...)
        .to_dataframe()[var_name]
        .unstack(level=2)
        .T
    )


def get_scalar_var(var_name: str, trace: az.InferenceData) -> pd.Series:
    """Extract chains/draws for a scalar variable.

    Returns Series of posterior samples.
    """
    return az.extract(trace, var_names=var_name).to_dataframe()[var_name]


def is_scalar_var(var_name: str, trace: az.InferenceData) -> bool:
    """Check if a variable in the trace is scalar (not a vector/time series).

    Returns True if the variable has only (chain, draw) dimensions,
    False if it has additional dimensions (e.g., time steps).
    """
    var_data = trace.posterior[var_name]
    # Scalar variables have only 'chain' and 'draw' dimensions
    return set(var_data.dims) == {"chain", "draw"}


def get_scalar_var_names(trace: az.InferenceData) -> list[str]:
    """Get list of all scalar variable names in the trace.

    Returns list of variable names that are scalars (not vectors).
    """
    return [
        var_name
        for var_name in trace.posterior.data_vars
        if is_scalar_var(var_name, trace)
    ]


def check_for_zero_coeffs(
    trace: az.InferenceData,
    critical_params: list[str] | None = None,
) -> pd.DataFrame:
    """Check scalar parameters for coefficients indistinguishable from zero.

    Automatically detects scalar variables (excludes vector/time series variables).
    Shows quantiles and flags parameters that may be indistinguishable from zero.

    Args:
        trace: InferenceData from model fitting
        critical_params: List of parameter names that are critical (warn if any
            quantile crosses zero). If None, uses default threshold of 2+ crossings.

    Returns:
        DataFrame with quantiles and significance markers.
    """
    from IPython.display import display

    if critical_params is None:
        critical_params = []

    q = [0.01, 0.05, 0.10, 0.25, 0.50]
    q_tail = [1 - x for x in q[:-1]][::-1]
    q = q + q_tail

    # Auto-detect scalar variables
    scalar_vars = get_scalar_var_names(trace)

    quantiles = {
        var_name: get_scalar_var(var_name, trace).quantile(q)
        for var_name in scalar_vars
    }

    df = pd.DataFrame(quantiles).T.sort_index()
    problem_intensity = (
        pd.DataFrame(np.sign(df.T))
        .apply([lambda x: x.lt(0).sum(), lambda x: x.ge(0).sum()])
        .min()
        .astype(int)
    )
    marker = pd.Series(["*"] * len(problem_intensity), index=problem_intensity.index)
    markers = (
        marker.str.repeat(problem_intensity).reindex(problem_intensity.index).fillna("")
    )
    df["Check Significance"] = markers

    for param in df.index:
        if param in problem_intensity:
            stars = problem_intensity[param]
            if (stars > 0 if param in critical_params else stars > 2):
                print(
                    f"*** WARNING: Parameter '{param}' may be indistinguishable from zero "
                    f"({stars} stars). Check model specification! ***"
                )

    print("=" * 20)

    return df


def _place_model_name(model_name: str, kwargs: dict) -> dict:
    """Place model_name in first available footer/header slot.

    Priority: rfooter, rheader, lheader (lfooter typically has content already).

    Args:
        model_name: Name of the model to place
        kwargs: Existing kwargs dict (will check for conflicts)

    Returns:
        Dict with the appropriate key set to model_name
    """
    # Try rfooter first, then rheader, then lheader
    for slot in ["rfooter", "rheader", "lheader"]:
        if slot not in kwargs:
            return {slot: model_name}
    # All slots taken - return empty (model name won't be shown)
    return {}


def plot_posteriors_kde(
    trace: az.InferenceData,
    model_name: str = "Model",
    **kwargs,
) -> None:
    """Plot separate Kernel Density Estimates for each coefficient posterior.

    Args:
        trace: InferenceData from model fitting
        model_name: Name of the model (placed in footer/header)
        **kwargs: Additional arguments passed to mg.finalise_plot()
    """
    scalar_vars = get_scalar_var_names(trace)

    for var_name in sorted(scalar_vars):
        samples = get_scalar_var(var_name, trace)

        _, ax = plt.subplots()

        # Plot KDE line
        samples.plot.kde(ax=ax, color="steelblue", linewidth=2)

        # Fill under the curve
        kde = stats.gaussian_kde(samples)
        x_range = np.linspace(samples.min(), samples.max(), 200)
        kde_values = kde(x_range)
        ax.fill_between(x_range, kde_values, alpha=0.3, color="steelblue")

        # Add vertical line at zero
        ax.axvline(x=0, color="darkred", linestyle="--", linewidth=1.5)

        # Add median line
        median_val = samples.quantile(0.5)
        ax.axvline(x=median_val, color="black", linestyle="--", linewidth=1, alpha=0.7)

        # Add median value text at top of curve
        max_y = kde_values.max()
        ax.text(
            median_val,
            max_y,
            f"{median_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )

        defaults = {
            "title": f"{var_name} Posterior",
            "xlabel": "Coefficient value",
            "lfooter": "Red dashed line marks zero. Black dashed line marks median.",
            **_place_model_name(model_name, kwargs),
        }
        for key in list(defaults.keys()):
            if key in kwargs:
                defaults.pop(key)

        mg.finalise_plot(
            ax,
            **defaults,
            **kwargs,
        )


def _auto_scale(samples: pd.Series, median: float) -> tuple[pd.Series, int]:
    """Scale samples for better visualization when values are large.

    Args:
        samples: Posterior samples
        median: Median of samples

    Returns:
        Tuple of (scaled samples, scale factor)
    """
    threshold = 1.3  # don't scale small values near 1
    if abs(median) <= threshold:
        return samples, 1
    scale = 10 ** math.floor(math.log10(abs(median * 10)))
    return samples / scale, max(int(scale), 1)


def plot_posteriors_bar(
    trace: az.InferenceData,
    model_name: str = "Model",
    **kwargs,
) -> None:
    """Plot horizontal bar chart of coefficient posteriors.

    Args:
        trace: InferenceData from model fitting
        model_name: Name of the model (placed in footer/header)
        **kwargs: Additional arguments passed to mg.finalise_plot()
    """
    scalar_vars = get_scalar_var_names(trace)

    posteriors = {}
    labels = {}
    all_significant_99 = True
    all_significant_95 = True

    for var in scalar_vars:
        samples = get_scalar_var(var, trace)
        median = samples.quantile(0.5)

        # Check 99% significance
        if median < 0:
            if samples.quantile(0.99) >= 0:
                all_significant_99 = False
            if samples.quantile(0.95) >= 0:
                all_significant_95 = False
        else:
            if samples.quantile(0.01) <= 0:
                all_significant_99 = False
            if samples.quantile(0.05) <= 0:
                all_significant_95 = False

        scaled_samples, scale = _auto_scale(samples, median)
        if scale != 1:
            posteriors[var] = scaled_samples
            labels[var] = f"{var}/{scale}"
        else:
            posteriors[var] = samples
            labels[var] = var

    cuts = [2.5, 16]
    palette = "Blues"
    cmap = plt.get_cmap(palette)
    color_fracs = [0.4, 0.7]

    _, ax = plt.subplots(figsize=(10, len(scalar_vars) * 0.6 + 1))

    y_positions = range(len(scalar_vars))
    bar_height = 0.7

    sorted_vars = sorted(scalar_vars)
    for i, var in enumerate(sorted_vars):
        samples = posteriors[var]

        for j, p in enumerate(cuts):
            quants = (p, 100 - p)
            lower = samples.quantile(quants[0] / 100.0)
            upper = samples.quantile(quants[1] / 100.0)
            height = bar_height * (1 - j * 0.25)

            ax.barh(
                i,
                width=upper - lower,
                left=lower,
                height=height,
                color=cmap(color_fracs[j]),
                alpha=0.7,
                label=f"{quants[1] - quants[0]:.0f}% HDI" if i == 0 else "_",
                zorder=j + 1,
            )

        median = samples.quantile(0.5)
        ax.vlines(
            median,
            i - bar_height / 2,
            i + bar_height / 2,
            color="black",
            linestyle="--",
            linewidth=1,
            zorder=10,
            label="Median" if i == 0 else "_",
        )
        ax.text(
            median,
            i + bar_height / 2 + 0.05,
            f"{median:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )

    ax.axvline(x=0, color="darkred", linestyle="--", linewidth=1.5, zorder=15)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([labels[var] for var in sorted_vars])
    ax.invert_yaxis()

    lfooter = "Some variables have been scaled (as indicated)."
    if all_significant_99:
        lfooter += " All coefficients are different from zero (>99% probability)."
    elif all_significant_95:
        lfooter += " All coefficients are different from zero (>95% probability)."

    defaults = {
        "title": "Coefficient Posteriors",
        "xlabel": "Coefficient value",
        "legend": {"loc": "best", "fontsize": "x-small"},
        "lfooter": lfooter,
        **_place_model_name(model_name, kwargs),
    }
    for key in list(defaults.keys()):
        if key in kwargs:
            defaults.pop(key)

    mg.finalise_plot(
        ax,
        **defaults,
        **kwargs,
    )


def posterior_predictive_checks(
    trace: az.InferenceData,
    model,
    obs_vars: dict[str, np.ndarray],
    obs_index: pd.Index,
    var_labels: dict[str, str] | None = None,
    model_name: str = "Model",
    **kwargs,
) -> az.InferenceData:
    """Generate and plot posterior predictive samples.

    Args:
        trace: InferenceData from model fitting
        model: PyMC model object
        obs_vars: Dictionary mapping observed variable names to their data arrays
        obs_index: Index for the time series (e.g., PeriodIndex)
        var_labels: Optional dictionary mapping variable names to display labels.
            If None, uses variable names as labels.
        model_name: Name of the model (placed in footer/header)
        **kwargs: Additional arguments passed to mg.finalise_plot()

    Returns:
        InferenceData with posterior predictive samples
    """
    import pymc as pm

    # Generate posterior predictive samples
    with model:
        ppc = pm.sample_posterior_predictive(trace, random_seed=42)

    if var_labels is None:
        var_labels = {k: k for k in obs_vars}

    for var_name, observed_data in obs_vars.items():
        # Extract posterior predictive samples
        ppc_samples = ppc.posterior_predictive[var_name].values
        # Shape: (chains, draws, time) -> flatten to (samples, time)
        ppc_flat = ppc_samples.reshape(-1, ppc_samples.shape[-1])

        ppc_mean = ppc_flat.mean(axis=0)
        ppc_05 = np.percentile(ppc_flat, 5, axis=0)
        ppc_95 = np.percentile(ppc_flat, 95, axis=0)
        ppc_16 = np.percentile(ppc_flat, 16, axis=0)
        ppc_84 = np.percentile(ppc_flat, 84, axis=0)

        # 90% CI band
        band_90 = pd.DataFrame({"lower": ppc_05, "upper": ppc_95}, index=obs_index)
        ax = mg.fill_between_plot(band_90, color="steelblue", alpha=0.15, label="90% CI")

        # 68% CI band
        band_68 = pd.DataFrame({"lower": ppc_16, "upper": ppc_84}, index=obs_index)
        ax = mg.fill_between_plot(
            band_68, ax=ax, color="steelblue", alpha=0.25, label="68% CI"
        )

        # Predicted mean
        predicted = pd.Series(ppc_mean, index=obs_index, name="Predicted mean")
        mg.line_plot(predicted, ax=ax, color="steelblue", width=1.5)

        # Observed
        observed = pd.Series(observed_data, index=obs_index, name="Observed")
        mg.line_plot(observed, ax=ax, color="darkred", width=1)

        label = var_labels.get(var_name, var_name)
        defaults = {
            "title": f"Posterior Predictive Check - {label}",
            "ylabel": label,
            "legend": {"loc": "upper right", "fontsize": "x-small"},
            "lfooter": "Blue: model prediction with credible intervals. Red: observed.",
            "y0": True,
            **_place_model_name(model_name, kwargs),
        }
        for key in list(defaults.keys()):
            if key in kwargs:
                defaults.pop(key)

        mg.finalise_plot(ax, **defaults, **kwargs)

    return ppc


def residual_autocorrelation_analysis(
    ppc: az.InferenceData,
    obs_vars: dict[str, np.ndarray],
    obs_index: pd.Index,
    var_labels: dict[str, str] | None = None,
    max_lags: int = 20,
    model_name: str = "Model",
    **kwargs,
) -> None:
    """Analyze residual autocorrelation for model validation.

    For state-space models, residuals should be approximately white noise
    (no significant autocorrelation).

    Args:
        ppc: InferenceData with posterior predictive samples
        obs_vars: Dictionary mapping observed variable names to their data arrays
        obs_index: Index for the time series (e.g., PeriodIndex)
        var_labels: Optional dictionary mapping variable names to display labels.
            If None, uses variable names as labels.
        max_lags: Maximum number of lags for ACF plot (default 20)
        model_name: Name of the model (placed in footer/header)
        **kwargs: Additional arguments passed to mg.finalise_plot()
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    if var_labels is None:
        var_labels = {k: k for k in obs_vars}

    for var_name, observed_data in obs_vars.items():
        # Get posterior predictive mean
        ppc_samples = ppc.posterior_predictive[var_name].values
        ppc_flat = ppc_samples.reshape(-1, ppc_samples.shape[-1])
        ppc_mean = ppc_flat.mean(axis=0)

        # Calculate residuals
        residuals = observed_data - ppc_mean
        residuals_series = pd.Series(residuals, index=obs_index, name="Residuals")

        # Plot residuals over time with ±2σ band
        std_band = pd.DataFrame(
            {
                "lower": np.full(len(obs_index), -2 * residuals.std()),
                "upper": np.full(len(obs_index), 2 * residuals.std()),
            },
            index=obs_index,
        )
        ax = mg.fill_between_plot(std_band, color="grey", alpha=0.1, label="±2σ")
        mg.line_plot(residuals_series, ax=ax, color="steelblue", width=0.8)

        # Ljung-Box test
        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        p_value = lb_test["lb_pvalue"].values[0]
        status = "OK" if p_value > 0.05 else "AUTOCORRELATED"

        label = var_labels.get(var_name, var_name)
        defaults = {
            "title": f"Residuals - {label}",
            "ylabel": "Residual",
            "lfooter": f"Ljung-Box test (lag 10): p={p_value:.4f} ({status})",
            "y0": True,
            **_place_model_name(model_name, kwargs),
        }
        for key in list(defaults.keys()):
            if key in kwargs:
                defaults.pop(key)

        mg.finalise_plot(ax, **defaults, **kwargs)

        # ACF plot
        n = len(residuals)
        acf_vals = np.correlate(
            residuals - residuals.mean(), residuals - residuals.mean(), mode="full"
        )
        acf_vals = acf_vals[n - 1 :] / acf_vals[n - 1]
        acf_series = pd.Series(
            acf_vals[: max_lags + 1], index=range(max_lags + 1), name="ACF"
        )

        # Confidence bounds
        conf_bound = 1.96 / np.sqrt(n)
        conf_band = pd.DataFrame(
            {
                "lower": np.full(max_lags + 1, -conf_bound),
                "upper": np.full(max_lags + 1, conf_bound),
            },
            index=range(max_lags + 1),
        )

        ax = mg.fill_between_plot(conf_band, color="red", alpha=0.1, label="95% CI")
        mg.line_plot(acf_series, ax=ax, color="steelblue", width=1.5)

        acf_defaults = {
            "title": f"Autocorrelation Function - {label}",
            "ylabel": "ACF",
            "xlabel": "Lag",
            "lfooter": "Red band: 95% confidence bounds for white noise.",
            "y0": True,
            **_place_model_name(model_name, kwargs),
        }
        for key in list(acf_defaults.keys()):
            if key in kwargs:
                acf_defaults.pop(key)

        mg.finalise_plot(ax, **acf_defaults, **kwargs)

    # Print summary
    print(f"\n{model_name}: Residual Autocorrelation Summary")
    print("-" * 50)
    for var_name, observed_data in obs_vars.items():
        ppc_samples = ppc.posterior_predictive[var_name].values
        ppc_mean = ppc_samples.reshape(-1, ppc_samples.shape[-1]).mean(axis=0)
        residuals = observed_data - ppc_mean
        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        p_value = lb_test["lb_pvalue"].values[0]
        status = "OK" if p_value > 0.05 else "*** AUTOCORRELATED ***"
        label = var_labels.get(var_name, var_name)
        print(f"{label:25} Ljung-Box p={p_value:.4f}  {status}")


def plot_timeseries(
    trace: az.InferenceData | None = None,
    var: str | None = None,
    index: pd.PeriodIndex | None = None,
    data: pd.DataFrame | None = None,
    legend_stem: str = "Model Estimate",
    color: str = "blue",
    start: pd.Period | None = None,
    cuts: Sequence[float] = (0.005, 0.025, 0.16),
    alphas: Sequence[float] = (0.1, 0.2, 0.3),
) -> Axes | None:
    """Plot time series with credible intervals.

    Can either fetch samples from trace using var name, or use pre-computed data.

    Args:
        trace: InferenceData object (required if data not provided)
        var: Variable name to extract from trace (required if data not provided)
        index: PeriodIndex for the time series (required if data not provided)
        data: Pre-computed samples DataFrame (rows=time, cols=samples)
        legend_stem: Label prefix for legend entries
        color: Color for bands and line
        start: Start period for plotting (None = use all data)
        cuts: Quantile cuts for confidence bands (must be between 0 and 0.5)
        alphas: Alpha values for each confidence band

    Returns:
        Matplotlib Axes object
    """
    if len(cuts) != len(alphas):
        raise ValueError("Cuts and alphas must have the same length")

    # Get samples either from trace or use provided data
    if data is not None:
        samples = data
    elif trace is not None and var is not None and index is not None:
        samples = get_vector_var(var, trace)
        samples.index = index
    else:
        raise ValueError("Must provide either (trace, var, index) or data")

    if start is not None:
        samples = samples[samples.index >= start]

    ax: Axes | None = None
    for cut, alpha in zip(cuts, alphas):
        if not (0 < cut < 0.5):
            raise ValueError("Cuts must be between 0 and 0.5")

        lower = samples.quantile(q=cut, axis=1)
        upper = samples.quantile(q=1 - cut, axis=1)
        band = pd.DataFrame({"lower": lower, "upper": upper}, index=samples.index)
        ax = mg.fill_between_plot(
            band,
            ax=ax,
            color=color,
            alpha=alpha,
            label=f"{legend_stem} {int((1 - 2 * cut) * 100)}% Credible Interval",
        )

    median = samples.quantile(q=0.5, axis=1)
    median.name = f"{legend_stem} Median"
    ax = mg.line_plot(median, ax=ax, color=color, width=1, annotate=True)

    return ax
