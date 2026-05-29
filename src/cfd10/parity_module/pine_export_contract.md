# Pine -> Python feature-parity export contract (SPX 1D)

## Why this file exists

`cfd10.parity_module.verify.verify_student_export` proves **internal parity**:
the rules emitted into `pine/cfd10_student.pine` reproduce the fitted sklearn
student tree's `predict()` **exactly** (0 disagreements on all 467,742 pooled
rows). That closes the *rules* gap.

The one gap it **cannot** close in Python is whether TradingView computes the
same **feature values** on-chart as the cfd10 Python feature bank
(`cfd10.feature_module`, `FeatureConfig()` defaults). TradingView and NumPy can
diverge on warm-up length, `ta.linreg` internals, `ta.atr` (Wilder RMA) seeding,
floating-point order, and how far back history is available
(`max_bars_back`). To pin that, the user must export TradingView's own per-bar
feature values and we diff them against the Python bank on the **same** SPX 1D
bars.

This document specifies **exactly** what to export.

## Symbol / timeframe (must match the Python side bit-for-bit)

- **Symbol**: `SPX` (the same series the cfd10 pool loads as `SPX` from
  `data/raw_v16`; use the identical TradingView feed you sourced the CSV from).
- **Timeframe**: `1D` (daily).
- **Chart history**: scroll/zoom so the chart has loaded the **full** daily
  history (the indicator sets `max_bars_back=5000`). Warm-up bars at the very
  start will not match (both sides emit NaN there); parity is asserted only on
  bars where **both** the Python feature and the Pine feature are defined.

## Step 1 — add Data-Window plots to the generated indicator

Open `pine/cfd10_student.pine` in the TradingView Pine editor and append the
block below (it references the `f_*` feature variables the generator already
computes). Each `plot(..., display=display.data_window)` surfaces one feature in
the Data Window / exported CSV under the given title.

```pine
// --- Parity export: feature values + bar clock (Data Window only) -----------
plot(time,  "bar_time_ms", display=display.data_window)   // bar open time, ms epoch (UTC)
plot(open,  "ohlc_open",   display=display.data_window)
plot(high,  "ohlc_high",   display=display.data_window)
plot(low,   "ohlc_low",    display=display.data_window)
plot(close, "ohlc_close",  display=display.data_window)
plot(volume,"ohlc_volume", display=display.data_window)

// The 15 features the two student trees split on (union of LOW + HIGH):
plot(f_price_return_L10,    "price_return_L10",    display=display.data_window)
plot(f_price_return_L20,    "price_return_L20",    display=display.data_window)
plot(f_price_return_L40,    "price_return_L40",    display=display.data_window)
plot(f_mom_divergence_L10,  "mom_divergence_L10",  display=display.data_window)
plot(f_mom_divergence_L20,  "mom_divergence_L20",  display=display.data_window)
plot(f_er_dir_p10,          "er_dir_p10",          display=display.data_window)
plot(f_er_dir_p40,          "er_dir_p40",          display=display.data_window)
plot(f_er_abs_p20,          "er_abs_p20",          display=display.data_window)
plot(f_sma_slope_s100,      "sma_slope_s100",      display=display.data_window)
plot(f_linreg_slope_s50,    "linreg_slope_s50",    display=display.data_window)
plot(f_linreg_slope_s100,   "linreg_slope_s100",   display=display.data_window)
plot(f_vola_pos_ATR_l14,    "vola_pos_ATR_l14",    display=display.data_window)
plot(f_vola_pos_StdDev_l30, "vola_pos_StdDev_l30", display=display.data_window)
plot(f_har_vol,             "har_vol",             display=display.data_window)
plot(f_pir_s100,            "pir_s100",            display=display.data_window)
```

(If you only need one side, the LOW tree uses 10 of these and the HIGH tree 11;
the union is the 15 listed. Exporting all 15 covers both sides.)

## Step 2 — export the Data Window to CSV

In TradingView: chart's overflow menu (`...`) on the indicator pane ->
**Export chart data...** -> choose **all available bars** -> CSV. The CSV must
contain **one row per daily bar** with these columns (exact header names as set
by the `title=` above):

| Column                | Meaning                                              |
| --------------------- | ---------------------------------------------------- |
| `time` (CSV bar time) | The chart's own ISO bar timestamp column.            |
| `bar_time_ms`         | Pine `time` (ms epoch, UTC) — the join key.          |
| `ohlc_open`           | `open`                                               |
| `ohlc_high`           | `high`                                               |
| `ohlc_low`            | `low`                                                |
| `ohlc_close`          | `close`                                              |
| `ohlc_volume`         | `volume`                                             |
| `price_return_L10`    | feature                                              |
| `price_return_L20`    | feature                                              |
| `price_return_L40`    | feature                                              |
| `mom_divergence_L10`  | feature                                              |
| `mom_divergence_L20`  | feature                                              |
| `er_dir_p10`          | feature                                              |
| `er_dir_p40`          | feature                                              |
| `er_abs_p20`          | feature                                              |
| `sma_slope_s100`      | feature                                              |
| `linreg_slope_s50`    | feature                                              |
| `linreg_slope_s100`   | feature                                              |
| `vola_pos_ATR_l14`    | feature                                              |
| `vola_pos_StdDev_l30` | feature                                              |
| `har_vol`             | feature                                              |
| `pir_s100`            | feature                                              |

Save it as `data/parity/spx_1d_pine_export.csv`.

## Step 3 — what we do with it

We rebuild the Python feature bank on the **same** SPX 1D OHLCV
(`build_feature_matrix(df, FeatureConfig())`), inner-join Pine vs Python on
`bar_time_ms` (the ms-epoch bar clock — the only unambiguous key; ISO strings can
differ by session/timezone), and report, per feature, on bars where both are
finite:

- max absolute difference and the bar at which it occurs,
- correlation, and
- the count of student-relevant **threshold crossings** that flip between Pine
  and Python (the only differences that can change a tree leaf, hence a signal).

The OHLCV columns are exported too so we can confirm the **inputs** match first
(if `ohlc_close` disagrees with the cfd10 `SPX` close, every downstream feature
will, and the fix is the data feed, not the math).

## Exact feature -> Pine semantics being pinned

All constants below are `FeatureConfig()` defaults baked into the generated
indicator (they are the cfd10 feature bank's, **not** the legacy
`speculatores_v15` preset inputs).

| Feature                 | Pine computation (in `cfd10_student.pine`)                                                              | Python reference                                        |
| ----------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `price_return_L{10,20,40}` | `nz((close - close[L]) / close[L])`                                                                  | `feature_module.momentum.price_return`                  |
| `mom_divergence_L{10,20}`  | `price_ret * vol_ret`, `vol_ret = (volume - volume[L]) / max(volume[L], 1)`                          | `feature_module.momentum.mom_divergence`                |
| `er_dir_p{10,40}`          | path `Sigma|close[i]-close[i+1]|`, net `close-close[P]`, `er = net/path` (0 if flat); directional    | `feature_module.efficiency.efficiency_ratio(dir=True)`  |
| `er_abs_p20`               | same path, net `|close-close[P]|`                                                                    | `feature_module.efficiency.efficiency_ratio(dir=False)` |
| `sma_slope_s100`           | `(sma_at(100,0)-sma_at(100,25))/(25*sma_at(100,0))*1000`; lag `d=max(round(S/4),2)=25`; SMA via `ta.cum` | `feature_module.trend.sma_slope`                    |
| `linreg_slope_s{50,100}`   | `(ta.linreg(close,S,0)-ta.linreg(close,S,1))/sma_at(S,0)*1000`                                       | `feature_module.trend.linreg_slope_norm`                |
| `vola_pos_ATR_l14`         | `pir_of(ta.atr(14), 100)`                                                                            | `feature_module.vola.vola_raw('ATR',14)`+`vola_position(.,100)` |
| `vola_pos_StdDev_l30`      | `pir_of(ta.stdev(close,30), 100)`                                                                    | `feature_module.vola.vola_raw('StdDev',30)`+`vola_position(.,100)` |
| `har_vol`                  | `calc_har_vol()` (Garman-Klass HAR norm, weights 0.36/0.28/0.28, windows 5/22)                       | `feature_module.garch_har.har_vol`                      |
| `pir_s100`                 | `pir_for_scale(100, max(100,20)=100)` (the `ta.cum` ratio-scan idiom)                                | `feature_module.sma_pir.pir_for_scale_series`           |

## Known, expected mismatch sources (not bugs)

- **Warm-up edges**: Python emits NaN on warm-up; Pine emits `0.0`/`0.5` via its
  ternary guards or `nz`. Excluded from parity by the both-finite mask.
- **`ta.linreg` / `ta.atr` history depth**: with too little loaded history (small
  `max_bars_back` or a short chart) TradingView truncates the window; load full
  history before exporting.
- **Float ordering**: cumulative-sum SMA (`ta.cum`) vs NumPy `cumsum` can differ
  by ~1e-9; only differences large enough to flip a `feature <= threshold` test
  matter, which is why Step 3 counts threshold crossings, not raw deltas.
