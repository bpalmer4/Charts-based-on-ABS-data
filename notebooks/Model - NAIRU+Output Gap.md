# NAIRU + Output Gap Model - Documentation

This document contains the detailed documentation for the NAIRU + Output Gap Model notebook.

---

## Acknowledgements

This work has drawn on ideas and approaches in the following:

NAIRU estimation:

 * https://www.rba.gov.au/publications/bulletin/2017/jun/pdf/bu-0617-2-estimating-the-nairu-and-the-unemployment-gap.pdf (Tom Cusbert)

 * https://treasury.gov.au/sites/default/files/2021-04/p2021-164397_estimatingthenairuinaustralia.pdf

 * https://github.com/MacroDave/NAIRU

Output gap and Okun's Law:

 * https://www.rba.gov.au/publications/rdp/2019/2019-07.html (RBA potential output estimates)

 * https://www.imf.org/external/pubs/ft/wp/2012/wp12256.pdf (IMF output gap estimation)

 * Ball, Leigh & Loungani (2017) - Okun's Law: Fit at 50?

---

## The PyMC Model

This model jointly estimates the NAIRU and potential output (output gap) using five equations. All GDP variables ($Y$, $Y^*$) are expressed in **log form** (specifically, $\ln(GDP) \times 100$), ensuring dimensional consistency throughout.

### 1. NAIRU State Space Model

$$ U^{*}_{t} = U^{*}_{t-1} + \epsilon_{U^*} $$

The NAIRU evolves as a random walk without drift.

### 2. Potential Output State Space Model (Cobb-Douglas Production Function)

$$ Y^{*}_{t} = Y^{*}_{t-1} + \alpha \cdot g^{K}_{t} + (1-\alpha) \cdot g^{L}_{t} + g^{MFP}_{t} + \epsilon_{Y^*} $$

Where:
- $Y^*_t$ is **log potential GDP** (scaled by 100)
- $g^{K}_{t}$ is quarterly capital stock growth (net capital stock of non-financial and financial corporations, Henderson smoothed)
- $g^{L}_{t}$ is quarterly labor force growth (population × participation rate), with Henderson MA substituted for COVID period
- $g^{MFP}_{t}$ is multi-factor productivity growth (see below)
- $\alpha$ is the capital share of income (~0.25-0.30, estimated by the model)

**Data sources:**
- Capital stock: ABS 1364.0.15.003 (Modellers Database) - "Non-financial and financial corporations; Net capital stock (Chain volume measures)"
- Labor force: ABS 1364.0.15.003 (Modellers Database) - "Total labour force"
- MFP: ABS 5204.0 Table 13 - "Multifactor productivity - Hours worked: Percentage changes"

**Multi-factor Productivity (MFP) construction:**

The MFP input is constructed as a hybrid series because ABS MFP data only begins ~1995:
- **Post-1995**: ABS MFP growth, smoothed with Henderson MA (25-term) to reduce year-to-year volatility
- **Pre-1995**: Labour productivity trend (GDP per hour worked) used as proxy

Annual MFP is converted to quarterly frequency (÷4) and interpolated. Mean annual MFP growth is ~0.7% (1995-2024), declining from ~2.3% in the late 1990s to ~0% currently, reflecting Australia's structural productivity slowdown.

**Production function interpretation:**

The Cobb-Douglas formulation decomposes potential GDP growth into three fundamental drivers:

1. **Capital accumulation** ($\alpha \cdot g^{K}_{t}$): Contribution from growth in the productive capital stock
2. **Labour force growth** ($(1-\alpha) \cdot g^{L}_{t}$): Contribution from available labor supply
3. **Multi-factor productivity** ($g^{MFP}_{t}$): Technological progress and efficiency gains

The production function: $Y = A \cdot K^\alpha \cdot L^{1-\alpha}$

In logs: $\ln(Y) = \ln(A) + \alpha \ln(K) + (1-\alpha) \ln(L)$

Growth rates: $g^Y = g^{MFP} + \alpha \cdot g^K + (1-\alpha) \cdot g^L$

This captures time-varying effects including:
- Structural productivity slowdown since the GFC
- Investment cycles and capital deepening
- Immigration waves and pauses (e.g., COVID border closures)
- Participation rate trends (rising female participation, aging workforce)

**Current estimates (as of late 2024):**
- Potential GDP growth: ~2.0% p.a. (consistent with RBA estimates)
- Capital share (α): ~0.25
- Trend decline: -0.04 percentage points per year

### 3. Price Inflation Equation (Expectations-Augmented Phillips Curve)

$$ \pi_{t} = \frac{\pi^e_t}{4} + \rho_{\pi}\Delta_4 \rho^{m}_{t-1} +
   \gamma_{\pi}\frac{(U_t - U^*_t)}{U_t} +
   \xi_{\pi}\Xi^2_{t-2} + \theta_{\pi}\omega_t + \epsilon_{\pi}$$

Where:
- $\pi_t$ is quarterly trimmed mean inflation
- $\pi^e_t$ is **anchored inflation expectations** (annual rate, divided by 4 for quarterly), constructed as:
  - Pre-1993: Adaptive expectations from Henderson-smoothed annual inflation (25-term)
  - 1993-1998: Linear phase-in of anchoring to 2.5% target
  - Post-1998: Fully anchored to 2.5% target
- $\Delta_4 \rho^{m}_{t-1}$ is the **four-quarter change in log import prices** (lagged), capturing external price shocks transmitted through the exchange rate
- $\Xi_{t-2}$ is the **Global Supply Chain Pressure Index** (NY Fed), with the squared term capturing asymmetric COVID-era supply disruptions (2020Q1-2023Q2)
- $\omega_t$ is the quarterly change in AUD-denominated oil prices, capturing energy price shocks
- $\gamma_{\pi}$ is the Phillips curve slope on the unemployment gap

**Inflation expectations construction:**

The expectations series combines historical headline CPI (Original series, back to 1949) with trimmed mean inflation (from 1987), applies Henderson smoothing, then phases in anchoring to reflect the RBA's inflation targeting credibility:
- Uses headline CPI for early period (pre-trimmed-mean era)
- Switches to trimmed mean when available (from 1987Q3)
- Henderson MA (25-term) smooths both series to capture long-memory expectations formation
- Phased anchoring reflects the gradual establishment of inflation targeting credibility after 1993

### 4. Okun's Law (change form)

$$ \Delta U_t = \beta_{okun}(Y_t - Y^{*}_t) + \epsilon_{okun} $$

This links the change in unemployment to the output gap. The coefficient $\beta_{okun}$ is expected to be negative: when output exceeds potential (positive output gap), unemployment falls.

This change form is more robust than the levels form, avoiding the need to relate two I(1) series directly.

### 5. Wage Growth Equation

$$ \Delta ulc_t = \alpha_{wg} + \gamma_{wg}\frac{(U_t - U^*_t)}{U_t} +
   \lambda_{wg}\frac{\Delta U_{t-1}}{U_t} + \epsilon_{wg}$$

Where $\Delta ulc_t$ is **quarterly growth in unit labour costs** (compensation per unit of output), used as a proxy for underlying wage inflation. The $\lambda_{wg}$ term captures "speed limit" effects where rapid changes in unemployment affect wage pressures beyond the level of the gap.

---

## Key Design Choices

1. **Log specification**: All GDP variables are in logs, so first-differences represent growth rates and the equations are dimensionally consistent.

2. **Joint estimation**: The NAIRU and potential output are estimated simultaneously, with Okun's Law providing the link between the two gaps.

3. **Cobb-Douglas production function**: Potential output growth is decomposed into capital, labor, and MFP contributions using a production function approach with estimated capital share (~0.25).

4. **Hybrid MFP series**: Combines ABS multi-factor productivity data (post-1995) with labour productivity proxy (pre-1995) to enable full sample estimation.

5. **Okun's Law in changes**: The relationship between unemployment changes and output gaps helps identify potential output while avoiding levels-on-levels regression issues.

6. **Output gap calculation**: After estimation, output gap = $(Y - Y^*)/Y^*$ as a percentage deviation from potential.

**Note on sample period:**

The model is estimated using data from 1984-Q3 onwards. When estimated using only post-1993 data (the inflation targeting era), the model exhibited divergences during sampling, suggesting the earlier data provides important information for identifying the latent states. The full sample is retained for model stability, though users should be aware that the pre-1993 period reflects a different monetary policy regime.

---

## Hypothesis Tests for Theoretical Expectations

Test whether key model parameters match their theoretical values:
- **α (alpha_capital)**: Expected ≈ 0.25-0.30 (capital share of income)
- **β (beta_okun)**: Expected < 0 (negative Okun coefficient)
- **γ_π (gamma_pi)**: Expected < 0 (negative Phillips curve slope)
- **γ_wg (gamma_wg)**: Expected < 0 (negative wage Phillips curve slope)

---

## Taylor Rule Assessment

The Taylor Rule provides a benchmark for monetary policy:

$$ i_t = r^* + \pi_{coef} \cdot \pi_t - 0.5\pi^* + 0.5 \cdot y_{gap} $$

Where:
- $i_t$ = prescribed nominal policy rate
- $r^*$ = neutral real rate (hybrid: 75% trend + 25% raw)
- $\pi_{coef}$ = time-varying inflation coefficient (1.6→1.2 over sample)
- $\pi_t$ = actual annual inflation
- $\pi^*$ = target inflation (2.5%)
- $y_{gap}$ = output gap (%)

**Choice of r\***: We use a hybrid approach:
- 75% from linear trend (fitted to raw potential growth, captures secular decline)
- 25% from raw model estimate (retains some cyclical signal)

**Time-varying inflation coefficient**: The response to inflation declines from 1.6 to 1.2 over the sample, reflecting:
- Early period: aggressive response needed to establish credibility
- Later period: anchored expectations + exchange rate channel allow lower coefficient
- Theoretical floor: must exceed 1.0 for stability (Taylor Principle)
- Endpoint coefficient (1.2) broadly consistent with RBA research estimates

**Caveats**:
- Policy rules like Taylor's are interpretive tools, not speed limits. They can illuminate whether policy is broadly tight or loose relative to historical norms, but they cannot capture the full information set, the non-linearities or the judgement calls that shape real-world decisions.
- Since the GFC, policy rates have run persistently below Taylor-rule benchmarks because the economic environment fundamentally changed. The neutral rate fell, inflation repeatedly undershot targets, and economies grappled with balance-sheet repair and chronic demand weakness. Central banks also adopted a risk-management mindset in the face of uncertainty and financial instability, placing greater weight on avoiding deflation and the lower bound. As a result, actual policy was looser than what a simple Taylor rule - estimated on pre-crisis dynamics - would prescribe.
- Model uncertainty is real (see those confidence bands).
