# Note: 2006→2011 dwelling-growth anomaly in the per-census CAGR chart

*Prepared 2026-06-13 in response to Ben Phillips (@BenPhillips_ANU) querying the 2011 dwelling-growth figure.*

**Short version:** the trough was a census **basis break**, not a data-entry error and not a
real construction collapse. ABS changed how the headline "occupied private dwellings" count
treats *visitor-only* and *other not-classifiable* households between 2006 and 2011. Putting
all censuses on a consistent (inclusive) basis removes the trough; the corrected 2006→2011
total-dwelling CAGR is **~1.6%/yr**, not 0.63%.

## 1. The figures are correctly transcribed
Both endpoints match ABS QuickStats exactly:

- 2006 occupied private dwellings = 7,596,183; all private dwellings = 8,426,559
- 2011 occupied = 7,760,320; unoccupied = 934,470 (occ+unocc = 8,694,790)

So the raw chart faithfully reproduced the published headlines.

## 2. The basis differs across years
The ABS methodology note is explicit on the 2011/2016/2021 QuickStats: dwelling counts
**"exclude visitor only and other non-classifiable households."** The 2006 (and earlier)
counts **include** them. Confirmed from the 2006 household composition:

| 2006 | count |
|---|---|
| Occupied private dwellings (headline, **inclusive**) | 7,596,183 |
| Family + lone-person + group (**classifiable only**) | 7,144,097 |
| Visitor-only + other-not-classifiable (vnc) | **451,086** |

So the chart compared an *inclusive* 2006 against an *exclusive* 2011 — mechanically crushing
the apparent growth. That single mismatched transition is the entire trough (1971–2006 are
mutually consistent on the inclusive basis; 2011–2021 are mutually consistent on the exclusive
basis, which is why only 2006→2011 misbehaves).

## 3. Harmonisation (inclusive basis)
"Occupied private dwelling" *counts* historically include vnc (an occupied dwelling is occupied
regardless of whether the household could be classified), so 1971–2006 are already inclusive.
We only add vnc back to 2011–2021, taken as the ABS "all private dwellings" header minus
(occupied + unoccupied):

| Year | occ+unocc (exclusive) | All private dwellings (inclusive) | vnc added |
|---|---|---|---|
| 2011 | 8,694,790 | 9,117,033 | 422,243 |
| 2016 | 9,325,947 | 9,901,496 | 575,549 |
| 2021 | 10,318,993 | 10,852,208 | 533,215 |

(2016's vnc is elevated — consistent with the 2016 online-form/non-response issues lifting
"not classifiable".)

## 4. Corrected per-census CAGR (total private dwellings, inclusive)

| Census | Original (mixed basis) | Corrected (inclusive) |
|---|---|---|
| 2001 | 1.66 | 1.66 |
| 2006 | 1.58 | 1.58 |
| **2011** | **0.63** | **1.59** |
| 2016 | 1.41 | 1.66 |
| 2021 | 2.04 | 1.85 |

Dwelling growth now eases smoothly from 2.66%/yr (1976) to ~1.6–1.85%/yr, with no
crash-and-spike.

## 5. The measure matters (total vs occupied)
Our chart uses **total** dwellings (occ+unocc); Ben's preferred measure is **occupied**. The
corrected 2006→2011 CAGR across the four combinations:

| Measure | Exclusive basis | Inclusive basis |
|---|---|---|
| Total dwellings | 1.74 | 1.59 |
| Occupied dwellings | 1.67 | 1.50 |

All land in **1.5–1.75%/yr**; the spread (~¼ pt) is the basis/measure sensitivity Ben flagged.

## 6. Cross-checks
- **Completions:** ~150k/yr over 2006–2011 ≈ 714k gross, ~640–690k net of demolitions. The
  inclusive census total grew 8,426,559 → 9,117,033 = **+690,474**. Consistent. The exclusive
  comparison implied only +268k, which never made sense.
- **Dwelling-stock series (ABS 6432.0):** on the inclusive basis the stock/census multiplier is
  **0.987 (2016) and 0.988 (2021)** — i.e. the census count and the independent dwelling-stock
  estimate agree to ~1% and the gap is stable. On the exclusive basis it was 1.048/1.039 (larger
  and drifting). The inclusive census is the better match to the physical stock.

## 7. Residual caveats (with total dwellings + ERP)
- **2021 unoccupied is COVID-inflated** (~1.04m; cf. *"Were there really 1 million unoccupied
  dwellings in Australia on census night 2021?"*, Australian Population Studies). This inflates
  the 2021 *total*, so the corrected 2016→2021 total-dwelling CAGR (1.85%) is probably a touch
  high, and the per-total-dwelling *level* dips at 2021. Using **occupied** dwellings (Ben's
  measure) removes this.
- **Numerator/denominator basis mismatch:** our ratio uses **ERP** (adults 18+, all usual
  residents incl. non-private dwellings) over a **census** dwelling count. Ben's construction
  (occupied dwellings + census population *in occupied dwellings* → average household size) is
  internally consistent census-on-census; ours mixes sources and won't match his levels exactly.

## Sources
- 2006 QuickStats: <https://www.abs.gov.au/census/find-census-data/quickstats/2006/0>
- 2011 QuickStats: <https://www.abs.gov.au/census/find-census-data/quickstats/2011/0>
- 2016 QuickStats: <https://www.abs.gov.au/census/find-census-data/quickstats/2016/0>
- 2021 QuickStats: <https://www.abs.gov.au/census/find-census-data/quickstats/2021/AUS>
- Dwelling stock: ABS 6432.0, table 643201 ("Number of residential dwellings ; Australia ;")
- 2021 unoccupied: Australian Population Studies — <https://australianpopulationstudies.org/index.php/aps/article/download/106/72>
