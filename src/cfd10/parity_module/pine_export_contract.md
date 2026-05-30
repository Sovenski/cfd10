# Pine -> Python feature-parity export contract (SPX 1D)

## Why this file exists

`cfd10.parity_module.verify.verify_student_export` proves **internal parity**:
the rules emitted into `pine/cfd10_student.pine` reproduce the fitted sklearn
student's `predict()` **exactly** (0 disagreements on all 467,742 pooled rows,
both sides) -- for a tree side via the nested `if/else` re-sim, for a gboost side
via the additive-margin re-sim `init + learning_rate * sum(stage leaf)` compared
to the baked `logit(threshold)`. That closes the *rules* gap.

The one gap it **cannot** close in Python is whether TradingView computes the
same **feature values** on-chart as the cfd10 Python feature bank
(`cfd10.feature_module`, `FeatureConfig(include_overextension=True)` -- the
45-feature +overext bank). TradingView and NumPy can diverge on warm-up length,
`ta.linreg` internals, `ta.atr` (Wilder RMA) seeding, `ta.stdev` / cumulative-sum
float order, the GJR-GARCH recursion seed, and how far back history is available
(`max_bars_back`). To pin that, the user must export TradingView's own per-bar
feature values and we diff them against the Python bank on the **same** SPX 1D
bars.

This document specifies **exactly** what to export.

## The deployed students (v2)

Both chosen students are single **decision trees** (`depth=8`, `<=64` leaves).
The export and parity code additionally support a **gboost** side (additive
margin `init + learning_rate * sum(stage leaf)`, decision `margin >=
logit(threshold)` in raw margin space -- no sigmoid is emitted); if a side is
re-distilled to a gboost, the same Data-Window export below still applies (only
the rule block in the `.pine` changes, not the feature set).

- **LOW** tree (bottom turns): 27 features.
- **HIGH** tree (top turns): 29 features.
- **Union** (what the indicator computes once): **39 features**.

## Symbol / timeframe (must match the Python side bit-for-bit)

- **Symbol**: `SPX` (the same series the cfd10 pool loads as `SPX` from
  `data/raw_v16`; use the identical TradingView feed you sourced the CSV from).
- **Timeframe**: `1D` (daily).
- **Chart history**: scroll/zoom so the chart has loaded the **full** daily
  history (the indicator sets `max_bars_back=5000`). Warm-up bars at the very
  start will not match (both sides emit NaN / a guard constant there); parity is
  asserted only on bars where **both** the Python feature and the Pine feature
  are defined.

## Step 1 -- add Data-Window plots to the generated indicator

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

// The 39 features the two student trees split on (union of LOW + HIGH).
// Momentum / efficiency / trend.
plot(f_price_return_L10,    "price_return_L10",    display=display.data_window)
plot(f_price_return_L20,    "price_return_L20",    display=display.data_window)
plot(f_price_return_L40,    "price_return_L40",    display=display.data_window)
plot(f_mom_divergence_L20,  "mom_divergence_L20",  display=display.data_window)
plot(f_mom_divergence_L40,  "mom_divergence_L40",  display=display.data_window)
plot(f_mom_velocity_L10,    "mom_velocity_L10",    display=display.data_window)
plot(f_mom_velocity_L20,    "mom_velocity_L20",    display=display.data_window)
plot(f_mom_velocity_L40,    "mom_velocity_L40",    display=display.data_window)
plot(f_er_dir_p10,          "er_dir_p10",          display=display.data_window)
plot(f_er_dir_p20,          "er_dir_p20",          display=display.data_window)
plot(f_er_dir_p40,          "er_dir_p40",          display=display.data_window)
plot(f_er_abs_p20,          "er_abs_p20",          display=display.data_window)
plot(f_er_abs_p40,          "er_abs_p40",          display=display.data_window)
plot(f_sma_slope_s20,       "sma_slope_s20",       display=display.data_window)
plot(f_sma_slope_s50,       "sma_slope_s50",       display=display.data_window)
plot(f_sma_slope_s100,      "sma_slope_s100",      display=display.data_window)
plot(f_linreg_slope_s20,    "linreg_slope_s20",    display=display.data_window)
plot(f_linreg_slope_s50,    "linreg_slope_s50",    display=display.data_window)
plot(f_linreg_slope_s100,   "linreg_slope_s100",   display=display.data_window)
// Volatility position.
plot(f_vola_pos_ATR_l30,      "vola_pos_ATR_l30",      display=display.data_window)
plot(f_vola_pos_StdDev_l14,   "vola_pos_StdDev_l14",   display=display.data_window)
plot(f_vola_pos_StdDev_l30,   "vola_pos_StdDev_l30",   display=display.data_window)
plot(f_vola_pos_Intraday_l14, "vola_pos_Intraday_l14", display=display.data_window)
plot(f_vola_pos_Intraday_l30, "vola_pos_Intraday_l30", display=display.data_window)
// PIR / agreement / GARCH-HAR.
plot(f_pir_s5,    "pir_s5",    display=display.data_window)
plot(f_pir_s10,   "pir_s10",   display=display.data_window)
plot(f_pir_s20,   "pir_s20",   display=display.data_window)
plot(f_pir_s50,   "pir_s50",   display=display.data_window)
plot(f_pir_s100,  "pir_s100",  display=display.data_window)
plot(f_agree_high,"agree_high",display=display.data_window)
plot(f_har_vol,   "har_vol",   display=display.data_window)
plot(f_gjr_asym,  "gjr_asym",  display=display.data_window)
// Overextension / vol-regime (top tells).
plot(f_dist_above_sma_z_l50,  "dist_above_sma_z_l50",  display=display.data_window)
plot(f_dist_above_sma_z_l100, "dist_above_sma_z_l100", display=display.data_window)
plot(f_dist_above_sma_z_l200, "dist_above_sma_z_l200", display=display.data_window)
plot(f_drawdown_from_high_l20,  "drawdown_from_high_l20",  display=display.data_window)
plot(f_drawdown_from_high_l100, "drawdown_from_high_l100", display=display.data_window)
plot(f_realized_vol_pct,  "realized_vol_pct",  display=display.data_window)
plot(f_up_streak_norm,    "up_streak_norm",    display=display.data_window)
```

(If you only need one side: the LOW tree uses 27 of these and the HIGH tree 29;
the union is the 39 listed. Exporting all 39 covers both sides.)

### Per-side feature membership (L = LOW tree, H = HIGH tree)

| Feature | L | H | | Feature | L | H |
| --- | :-: | :-: | --- | --- | :-: | :-: |
| `price_return_L10`      | L | H | | `pir_s5`                | L | . |
| `price_return_L20`      | . | H | | `pir_s10`               | L | . |
| `price_return_L40`      | . | H | | `pir_s20`               | L | H |
| `mom_divergence_L20`    | . | H | | `pir_s50`               | L | H |
| `mom_divergence_L40`    | L | H | | `pir_s100`              | L | . |
| `mom_velocity_L10`      | L | . | | `agree_high`            | . | H |
| `mom_velocity_L20`      | L | H | | `har_vol`               | L | H |
| `mom_velocity_L40`      | . | H | | `gjr_asym`              | . | H |
| `er_dir_p10`            | L | . | | `dist_above_sma_z_l50`  | L | H |
| `er_dir_p20`            | . | H | | `dist_above_sma_z_l100` | L | H |
| `er_dir_p40`            | L | . | | `dist_above_sma_z_l200` | L | H |
| `er_abs_p20`            | . | H | | `drawdown_from_high_l20`  | L | . |
| `er_abs_p40`            | L | H | | `drawdown_from_high_l100` | L | H |
| `sma_slope_s20`         | L | H | | `realized_vol_pct`      | L | . |
| `sma_slope_s50`         | L | H | | `up_streak_norm`        | . | H |
| `sma_slope_s100`        | L | H | | `vola_pos_ATR_l30`      | L | H |
| `linreg_slope_s20`      | L | . | | `vola_pos_StdDev_l14`   | . | H |
| `linreg_slope_s50`      | L | . | | `vola_pos_StdDev_l30`   | L | H |
| `linreg_slope_s100`     | L | H | | `vola_pos_Intraday_l14` | . | H |
|                         |   |   | | `vola_pos_Intraday_l30` | . | H |

## Step 2 -- export the Data Window to CSV

In TradingView: chart's overflow menu (`...`) on the indicator pane ->
**Export chart data...** -> choose **all available bars** -> CSV. The CSV must
contain **one row per daily bar** with these columns (exact header names as set
by the `title=` above):

| Column                | Meaning                                              |
| --------------------- | ---------------------------------------------------- |
| `time` (CSV bar time) | The chart's own ISO bar timestamp column.            |
| `bar_time_ms`         | Pine `time` (ms epoch, UTC) -- the join key.         |
| `ohlc_open`           | `open`                                               |
| `ohlc_high`           | `high`                                               |
| `ohlc_low`            | `low`                                                |
| `ohlc_close`          | `close`                                              |
| `ohlc_volume`         | `volume`                                             |
| the 39 feature titles | one column each, exactly as named in Step 1.         |

Save it as `data/parity/spx_1d_pine_export.csv`.

## Step 3 -- what we do with it

We rebuild the Python feature bank on the **same** SPX 1D OHLCV
(`build_feature_matrix(df, FeatureConfig(include_overextension=True))`),
inner-join Pine vs Python on `bar_time_ms` (the ms-epoch bar clock -- the only
unambiguous key; ISO strings can differ by session/timezone), and report, per
feature, on bars where both are finite:

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
`speculatores_v15` preset inputs). The SMA idiom is the stateless `ta.cum`
`sma_at` (Pine L87-93); `pir_of` / `pir_for_scale` are L68-71 / L98-126.

| Feature family | Pine computation (in `cfd10_student.pine`) | Python reference |
| --- | --- | --- |
| `price_return_L{10,20,40}`  | `nz((close - close[L]) / close[L])`                                                       | `feature_module.momentum.price_return`                  |
| `mom_divergence_L{20,40}`   | `price_ret * vol_ret`, `vol_ret = (volume - volume[L]) / max(volume[L], 1)`               | `feature_module.momentum.mom_divergence`                |
| `mom_velocity_L{10,20,40}`  | `nz(price_ret - price_ret[1])` (bar-over-bar change of `price_return_L`)                  | `feature_module.momentum.mom_velocity`                  |
| `er_dir_p{10,20,40}`        | path `Sigma|close[i]-close[i+1]|`, net `close-close[P]`, `er = net/path` (0 if flat)      | `feature_module.efficiency.efficiency_ratio(dir=True)`  |
| `er_abs_p{20,40}`           | same path, net `|close-close[P]|`                                                         | `feature_module.efficiency.efficiency_ratio(dir=False)` |
| `sma_slope_s{20,50,100}`    | `(sma_at(S,0)-sma_at(S,d))/(d*sma_at(S,0))*1000`; lag `d=max(round(S/4),2)`               | `feature_module.trend.sma_slope`                        |
| `linreg_slope_s{20,50,100}` | `(ta.linreg(close,S,0)-ta.linreg(close,S,1))/sma_at(S,0)*1000`                            | `feature_module.trend.linreg_slope_norm`                |
| `vola_pos_ATR_l{30}`        | `pir_of(ta.atr(L), 100)`                                                                  | `vola.vola_raw('ATR',L)` + `vola_position(.,100)`       |
| `vola_pos_StdDev_l{14,30}`  | `pir_of(ta.stdev(close,L), 100)`                                                          | `vola.vola_raw('StdDev',L)` + `vola_position(.,100)`    |
| `vola_pos_Intraday_l{14,30}`| `pir_of(ta.sma(close>0?(high-low)/close:0, L), 100)`                                      | `vola.vola_raw('Intraday',L)` + `vola_position(.,100)`  |
| `pir_s{5,10,20,50,100}`     | `pir_for_scale(S, max(S,20))` (the `ta.cum` ratio-scan idiom)                             | `feature_module.sma_pir.pir_for_scale_series`           |
| `agree_high`                | fraction of scales `s in range(3,121,13)` with `pir_for_scale(s,max(s,20)) > 0.8`         | `feature_module.bank._block_agreement` (`agree_high`)   |
| `har_vol`                   | `calc_har_vol()` (Garman-Klass HAR norm, weights 0.36/0.28/0.28, windows 5/22)            | `feature_module.garch_har.har_vol`                      |
| `gjr_asym`                  | `calc_gjr_asym()` (GJR-GARCH(1,1) leverage asym, alpha/beta/gamma 0.03/0.90/0.08, SMA 252)| `feature_module.garch_har.gjr_asym`                     |
| `dist_above_sma_z_l{50,100,200}` | `tanh((close - sma_at(L,0)) / ta.stdev(close,100))` (flat std -> 0)                  | `feature_module.overextension.dist_above_sma_z` (z_win=100) |
| `drawdown_from_high_l{20,100}`   | `(close - ta.highest(close,L)) / ta.highest(close,L)` (non-positive high -> 0)      | `feature_module.overextension.drawdown_from_high`       |
| `realized_vol_pct`          | `pir_of(ta.stdev(log(close/close[1]), 20), 100)`                                          | `feature_module.overextension.realized_vol_pct` (win=20, range=100) |
| `up_streak_norm`            | `min(consecutive-up-bar streak, 5) / 5` (streak reset on a down/flat bar)                 | `feature_module.overextension.up_streak_norm` (cap=5)   |

`tanh(x)` is emitted as `(exp(2x)-1)/(exp(2x)+1)` (Pine has no `math.tanh`); this
is the squashing used by `dist_above_sma_z`, NOT a decision sigmoid.

## Gboost decision (only if a side is re-distilled to a gboost)

A gboost side emits, in pure Pine float arithmetic, one nested `if/else` per
stage returning that stage's raw leaf value, then

```
margin = init + learning_rate * (stage_0 + stage_1 + ...)
signal = margin >= logit(threshold)      // threshold default 0.5 -> logit = 0.0
```

No sigmoid is emitted: since the logistic link is monotone, `proba >= threshold`
is identical to `margin >= logit(threshold)`, so the call is exact float
arithmetic. `verify_student_export` re-implements this margin from the JSON and
asserts the binary signal matches `GradientBoostingClassifier.predict()` exactly,
reporting the margin max-abs error (~0 / 1e-9; the only slack is the JSON's 8-dp
rounding of `init`).

## Known, expected mismatch sources (not bugs)

- **Warm-up edges**: Python emits NaN on warm-up; Pine emits `0.0` / `0.5` /
  `float(na)` via its ternary guards or `nz`. Excluded from parity by the
  both-finite mask. (Notably `dist_above_sma_z`, `gjr_asym` and `har_vol` carry
  long warm-ups: 100, 252 and 22 bars respectively.)
- **`ta.linreg` / `ta.atr` / `ta.stdev` history depth**: with too little loaded
  history (small `max_bars_back` or a short chart) TradingView truncates the
  window; load full history before exporting.
- **GJR-GARCH recursion**: the `var float` state seeds from the long-run variance
  on bar 0 exactly as the Python port does, but the 252-bar SMA warm-up means the
  output is `na` until bar 251 on both sides.
- **Float ordering**: cumulative-sum SMA (`ta.cum`) vs NumPy `cumsum`, and the
  GK / HAR / GJR float reductions, can differ by ~1e-9; only differences large
  enough to flip a `feature <= threshold` test matter, which is why Step 3 counts
  threshold crossings, not raw deltas.
