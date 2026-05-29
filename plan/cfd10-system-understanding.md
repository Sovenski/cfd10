# cfd10 — Speculatores System Understanding

> A synthesis of seven examiner reports covering the Pine Script v6 detector, its preset/optimizer provenance, the Pine↔Python parity contract, and the multi-asset data corpus.
>
> Primary artifact: `C:\Users\kuben\Desktop\Projekte\cfd10\pine\speculatores_v15_presets_gold.pine` (662 lines).
> Line citations below refer to that file unless another path is given.

---

## 1. Executive Summary

**cfd10 is a market top/bottom (swing-turn) detection system built around a single Pine Script v6 indicator whose entire behavior is driven by externally-optimized parameter bundles.** The Pine file (`pine/speculatores_v15_presets_gold.pine`, titled *"Speculatores V15 Presets - Per-Side Regimes (edge voting)"*) is an `overlay=true` indicator that, bar by bar, decides whether the current bar looks like a market **top** (plots a red down-triangle) or a **bottom** (plots a green up-triangle). It does this through two fully independent, mirror-image pipelines — a HIGH side and a LOW side — that share no state and no parameters.

The defining engineering characteristic of this project is **a strict Pine↔Python numerical-parity discipline.** The detector logic is implemented twice: once in Pine (what renders on a TradingView chart) and once in an external Python pipeline (the optimizer + a feature-enrichment reference). The two must agree to the digit. A dedicated "parity shim" inside the Pine file (lines 73–113) exists solely because Pine's stateful `ta.*` built-ins corrupt their internal history buffers when called inside a variable-length loop; the shim replaces them with a stateless cumulative-sum trick. A large block of ~30 `dbg_*` data-window exports (lines 628–658) exists *only* to be diffed bar-by-bar against the Python reference table (`data/enriched/SPX_1D_18710201_20260318_TV_V11.csv`).

Parameters are not hand-set. They are produced by an **Optuna-style Python optimizer** that runs trials across folds, scores candidates with evolving scorers (Path A → Scorer v3 → Scorer v4), and emits the winning HIGH-side and winning LOW-side parameter vectors *separately*. Those winners are frozen into **14 named presets** selected by one dropdown. Some presets are trusted production winners; others are explicitly kept on the chart as "noise floor" or "likely outlier" diagnostics for visual comparison only.

The system also carries a **multi-asset TradingView data corpus** in three tiers: `data/raw` (14 hand-picked CSVs), `data/raw_v16` (80 CSVs — the V16-era bulk multi-asset × multi-timeframe export), and `data/enriched` (the Python ground-truth feature dump used for parity). The corpus spans indices, commodity futures, FX, and ~28 mega-cap single stocks, with daily history reaching back to the reconstructed 1871 S&P composite.

---

## 2. The Detection Algorithm

Every detection quantity exists as an independent `_high` (top) and `_low` (bottom) variable, with its own preset-tuned parameter and its own `var` state. The two pipelines **never gate each other** — a bar can be a top, a bottom, both, or neither. The final signals are (lines 585–586):

```pine
pivot_high = gate_pass_high and ph_confirms >= high_required and bars_since_high_signal > cooldown_bars_high
pivot_low  = gate_pass_low  and pl_confirms >= low_required  and bars_since_low_signal  > cooldown_bars_low
```

So a signal fires **iff** three things hold simultaneously: the **hard gate** passes (an AND of necessary conditions), the **soft-vote count** reaches a per-side required threshold, and the **cooldown** has elapsed. The end-to-end pipeline is: *multi-scale PIR agreement → hard gates → soft confirmation votes → cooldown.* Concrete numbers below are for the default preset, **"Gold 1D Current"** (`is_gold_current`, line 191).

### 2.1 The core primitive: PIR (position-in-range)

`pir_of(val, lookback)` (lines 68–71) returns where `val` sits inside its own [min, max] over `lookback` bars: `(val − lo)/(hi − lo)`, or `0.5` if flat. Output ∈ [0,1]; ≈1.0 = top of its recent range, ≈0.0 = bottom. Almost everywhere, `val` is the ratio `close / SMA(close, s)` — the *stretch* of price above/below an s-bar moving average. PIR ≈ 1 means price is the most over-extended above that MA it has been in `lookback` bars (topping); PIR ≈ 0 means maximally below (bottoming).

### 2.2 Multi-scale PIR agreement (the consensus across time-scales)

`calc_agreement(scale_start, scale_end, scale_step, pct_extreme)` (lines 115–126) loops a scale `s` from start to end by step. For each scale it computes `pir_for_scale(s, max(s,20))` (the loop-safe PIR, §5.2) and counts:

- `scales_high += 1` if `pir_s > pct_extreme` (price near the TOP of its s-bar range)
- `scales_low  += 1` if `pir_s < (1 − pct_extreme)` (price near the BOTTOM)

It returns the counts and the fractions `agreement_high = scales_high/n_scales` and `agreement_low = scales_low/n_scales`. The HIGH side consumes `agreement_high_high`; the LOW side consumes `agreement_low_low` (lines 303–304). For Gold, the HIGH window is `scale_start=3 → scale_end=270 step 3` (90 scales) with `pct_extreme_high = 0.96` (line 218); the LOW window has its own start/end/step with `pct_extreme_low = 0.85` (line 257).

> **SPX override (not Gold):** For three SPX presets only, the loop result is *overwritten* by a hand-unrolled sum of `low_scale_flag(...)` / `high_scale_flag(...)` over fixed scale lists (lines 306–387) — 26 low scales, 45 high scales. Those flags call the *stateful* `ta.sma` but are safe because each is a distinct fixed call-site, not a loop. For Gold these `if` blocks are skipped entirely.

### 2.3 The hard gate stack (ALL must be true)

`gate_pass_high` / `gate_pass_low` (lines 576–577) are a logical AND of necessary conditions — the must-pass eligibility test. Miss any one and there is no signal, regardless of votes.

```pine
gate_pass_high = price_high_ok and ms_agree_high and dur_high_flag
                 and not scale_div_high_flag and er_gate_ok_high
                 and (not use_pivot_drift_high or not pivot_drift_gate_up_high)   // 576
gate_pass_low  = price_low_ok  and ms_agree_low  and dur_low_flag
                 and not scale_div_low_flag  and er_gate_ok_low                    // 577
```

The hard-gate components:

| Gate | Lines | Rule | Financial reading |
|---|---|---|---|
| **Price extreme** | 465–466 | `high ≥ ta.highest(high, lb)[1]` (LOW: `low ≤ ta.lowest(low, lb)[1]`) | A fresh local high/low must have just printed. Disabled toggle passes trivially. |
| **ms_agree** | 389–390 | `agreement_*_* ≥ min_agreement_*` | Enough time-scales must agree price is stretched. Gold: 0.75 of HIGH scales, only 0.30 of LOW scales (lines 219, 258). |
| **Duration-at-extreme** | 415–436 | `dur_at_* ≥ min_duration_*` | The extreme must have *persisted*. Gold HIGH needs 13 consecutive bars (line 213). |
| **Scale-divergence veto** | 392–400 | `not scale_div_*_flag` | The single detection-scale extremity must AGREE with the broad multi-scale fraction. |
| **Efficiency-ratio gate** | 468–480 | `er_val < nz(er_val[1], 0)` | The Kaufman ER must be FALLING — trend cleanliness deteriorating into a turn. |
| **Pivot-drift veto (HIGH only)** | 503, 576 | `not pivot_drift_gate_up_high` | Do NOT call a top while confirmed pivots are drifting *strongly* up (Gold gate mult = 8.0, line 229). |

**The scale-divergence veto** (lines 392–400) deserves emphasis. A single "detection-scale" SMA (`S_detect_high = 12` for Gold, line 209) yields `pir_detect_high`. Then `scale_div_high = pir_detect_high − agreement_high_high` (line 397); the LOW side uses `(1 − pir_detect_low) − agreement_low_low` (line 398) to align orientation. The flag fires when `abs(...) > thresh` (Gold: 0.35 high, 0.39 low; lines 223, 262). It is precisely a consistency check between a *narrow* lookback (one detect scale) and a *wide* lookback (the 3..270 scale band).

**The duration state machine** (lines 415–436) carries a one-bar hysteresis: `dur_at_*` increments while `agreement > dur_extreme_pct_*`, and only resets after MORE THAN ONE consecutive miss (`dur_miss > 1`, lines 423/434). Note `dur_extreme_pct_*` is a *separate, stricter-context* threshold than `min_agreement_*` (Gold HIGH: 0.83 for duration vs 0.75 for ms_agree).

### 2.4 The soft confirmation votes (a COUNT, not all-must-pass)

This is the crisp contrast to the hard gate. `ph_confirms` / `pl_confirms` (lines 550–568) **sum up to eight** boolean confirmations, each guarded by its own `use_*` toggle (HIGH list, lines 550–558):

```
ph_confirms += use_trend_high            and t_up_h_eff    (trend up)
            +  use_pivot_drift_high       and pd_dn_h_eff   (pivot drift down)
            +  use_volume_high            and vs_h_eff      (volume drying up)
            +  use_momentum_high          and md_h_eff      (price·volume divergence < 0)
            +  use_momentum_velocity_high and mv_h_eff      (momentum decelerating)
            +  use_volatility_high        and va_h_eff      (volatility elevated)
            +  use_gjr_asym_high          and g_h_eff       (GJR asymmetry vote)
            +  use_har_vol_high           and h_h_eff       (HAR vol vote)
```

The signal requires `ph_confirms ≥ high_required` — **at least `required` of them, not all.** The required count (lines 573–574) is the base `confirm_count_*` (Gold: 5 high, 3 low; lines 221, 260), clamped to `[1, max_votes]`, with a pivot-drift confirm-bias adjustment (§2.6).

The `*_eff` bools (lines 530–546) are the underlying indicators optionally wrapped by the V15 **edge-voting** layer (§5.4). For Gold the edge toggles default to false, so `*_eff == raw state` and behavior matches V14.

### 2.5 Cooldown throttle

Persistent bar counters throttle repeat fires (lines 580–591). `bars_since_*_signal` initializes to **999** (so the first eligible bar can fire), increments each bar, and resets to 0 on a fire. The signal additionally requires `bars_since_*_signal > cooldown_bars_*` (Gold HIGH = 9, line 214). This is the final throttle, applied *after* both the hard gate and the soft-vote test pass.

### 2.6 The two deliberate HIGH/LOW asymmetries

The pipelines are mirror images with exactly **two** structural asymmetries — both easy to misread:

1. **Extra pivot-drift veto on HIGH only.** `gate_pass_high` carries `not pivot_drift_gate_up_high`; `gate_pass_low` has no symmetric down-drift veto. Tops are suppressed during a strong up-drift; bottoms are *not* symmetrically suppressed during a down-drift.
2. **Opposite-sign confirm-bias (lines 573–574).** *Both* sides test the same condition (`pivot_drift_up_*`, i.e. drift UP) but move the required count in opposite directions: HIGH **raises** `high_required` by `+pivot_drift_confirm_bias_high` (tops harder to confirm during an up-drift), while LOW **lowers** `low_required` by `−pivot_drift_confirm_bias_low` (a dip in an uptrend is a more likely bottom). For Gold the LOW bias is 0 (line 269); the HIGH bias falls through to the trailing `:1` (an open question, §8).

### Hard vs Soft — the crisp rule

- **HARD GATE** (`gate_pass_*`, AND of all): price extreme, ms_agree, duration, scale-divergence-not-vetoed, ER-falling, and (HIGH only) pivot-drift-not-strongly-up. *Every one must be true.*
- **SOFT VOTES** (`*_confirms ≥ *_required`, a COUNT): trend, volume, momentum, momentum-velocity, volatility, GJR, HAR (+pivot-drift contributes to the count). *Need at least `required` of them.*
- **FINAL**: hard gate AND soft-count AND cooldown → signal.

---

## 3. Indicator / Feature Catalog

Each feature collapses to one boolean **vote** about whether the current bar looks like a top (HIGH) or bottom (LOW). Three features are **gates** (veto) rather than votes — they are necessary conditions, not part of the tunable quorum.

| # | Feature | Lines | Formula (essence) | Role | Vote direction / financial meaning |
|---|---|---|---|---|---|
| 1 | **PIR** `pir_of` / `pir_for_scale` | 68–71 / 98–113 | min-max position of `close/SMA(s)` ratio ∈ [0,1] | primitive | ≈1 over-extended above MA (top); ≈0 below MA (bottom). |
| 2 | **Multi-scale trend** | 402–413 | `slope_val = (SMA−SMA[Δ])/(Δ·SMA)·1000` AND `linreg_norm = (linreg₀−linreg₁)/SMA·1000` | VOTE | HIGH `trend_up`: both estimators > thresh (uptrend = context to top from). LOW `trend_down`: both < −thresh (downtrend precondition for a bottom). **Requires BOTH** estimators (noise filter). |
| 3 | **Duration-at-extreme** | 415–436 | consecutive bars `agreement > dur_extreme_pct`, resets after 2 misses | **GATE** | Persistence filter; a turn must be sustained ≥ `min_duration` bars. |
| 4 | **Volume surge** | 439–440, 524–525 | `SMA(vol, fast)/SMA(vol, slow)` | VOTE | **Asymmetric by design.** LOW votes on a *surge* (`> thresh`, capitulation/climax). HIGH votes on the **reciprocal** (`< 1/thresh`, volume *drying up* — exhaustion at a top). |
| 5 | **Momentum divergence** | 443–445, 449–451, 526–527 | `price_ret × vol_ret` (product of N-bar returns) | VOTE | Both sides vote when product `< 0` — price and volume moving *opposite* ways = unsupported move, reversal-suggestive either direction. |
| 6 | **Momentum velocity** | 446–447, 452–453 | `price_ret − price_ret[1]` (price acceleration) | VOTE | "Reversal" mode → decelerating into the turn (exhaustion). "Trend" mode → still accelerating. LOW flips every comparison sign. Gold uses Reversal both sides (line 244). |
| 7 | **Volatility elevated** | 455–462 | `pir_of(vola_raw, range_len)`; `vola_raw` = ATR / `(H−L)/C` / StdDev per `vola_method` | VOTE | Both sides vote when `vola_pos > vola_high_pct` (same direction) — high vol accompanies blow-off tops AND capitulation bottoms. |
| 8 | **Efficiency-ratio (Kaufman)** | 468–480 | `er_val = net_move / Σ|bar-to-bar move|` over `er_period` | **GATE** | Passes only when ER is **falling** (`er_val < er_val[1]`) — trend losing cleanliness into a reversal. `er_directional` toggles signed [−1,1] vs absolute [0,1]. |
| 9 | **GJR-GARCH asymmetry** | 159–177, 483–485 | leverage-aware GARCH variance ÷ symmetric counterpart; `α=0.03, β=0.90, γ=0.08` | VOTE | **Sign discriminates side.** HIGH votes on `norm ≤ −thresh` (upside-driven vol). LOW votes on `norm ≥ +thresh` (downside leverage / capitulation). The `γ` term weights `r²` only after down days (line 172). |
| 10 | **HAR + Garman-Klass vol** | 179–188, 486–488 | GK OHLC variance → HAR mix `0.36·daily + 0.28·weekly + 0.28·monthly`; forecast/spot ratio | VOTE | **Both sides test the same direction** (`norm ≥ thresh`) — expanding/long-memory volatility is reversal-suggestive either way; only the threshold differs per side. |
| 11 | **Pivot drift** | 142–151, 491–504 | avg %-change per confirmed pivot over `lookback` confirmed pivots | VOTE + **GATE** + bias | **Three roles.** Soft vote: HIGH `drift_down` (structure rolling over → top), LOW `drift_up` (structure turning up → bottom). Hard veto: `pivot_drift_gate_up_high` forbids a top in a strong uptrend. Bias: nudges `*_required` (§2.6). |
| 12 | **Baseline pivots** | 491–492 | `ta.pivothigh/low(., 20, 20)` (`baseline_lb=20`) | feeder | Standard fractal pivots; lag 20 bars (plotted `offset=−20`). Don't vote directly — they feed `confirmed_pivots`, the raw material for pivot drift. |
| 13 | **Edge-voting wrapper** | 506–547 | `edge_or_state(v, win, use_edge) = use_edge ? (v and not v[win]) : v` | vote modifier | V15 layer: converts "true now" (state) into "just turned true within `win` bars" (rising edge), killing burst-cluster false tops in sustained rallies. Off for Gold (= V14 state). |

**Critical no-lookahead detail:** pivot drift is read at the top of the bar (lines 495–496) *before* the current bar's confirmed pivot is pushed into `confirmed_pivots` (lines 497–500). The ordering is load-bearing; reversing it would introduce lookahead.

**Dead/legacy parameters:** `vola_low_pct` (lines 226, 265) and `vola_slope_lb` (lines 293, 300) are defined per-preset but are NOT consumed by any vote/gate in the shown code — only `vola_high_pct` drives `vola_elevated`. Likely vestigial from V11.

---

## 4. Preset & Optimizer Architecture

### 4.1 What a "preset" is, mechanically

A preset is **not** a config object — it is a *string label* (line 61, a 14-option dropdown) expanded into 14 mutually-exclusive boolean flags `is_gold_current … is_legacy` (lines 191–204). Every tunable parameter is then a standalone nested ternary `var = is_A ? valA : is_B ? valB : … : default`, **declared twice** — once for the HIGH side and once for the LOW side. Example pairs: `S_detect_high` (209) vs `S_detect_low` (248); `min_agreement_high` (219) vs `min_agreement_low` (258); `vola_method_high` (246) vs `vola_method_low` (285).

A single preset therefore freezes **~30 HIGH parameters and ~30 LOW parameters** simultaneously, and the two sides are fully decoupled — the optimizer found the best top-detector and the best bottom-detector as *separate searches*. This is why the header reports a distinct trial number and score for each side (e.g. Gold 1D Current: HIGH = trial 292 score 0.3337, LOW = trial 465 score 0.4078; header lines 12–13).

Derived lookbacks (lines 288+) are computed *from* `S_detect_*` (e.g. `slope_delta = max(round(S/4), 2)`, `vol_fast_len`, `mom_lookback = S_detect`), so the detection scale propagates into the slope/volume/momentum windows rather than being independently tuned.

> **Legacy fallthrough gotcha:** `is_legacy` is *never tested* in any parameter ternary. "Legacy V11 Gold" is encoded purely as the final fallthrough default after the last colon — selecting it lands on every default arm (e.g. `S_detect_high=22`, `min_agreement_high=0.30`). Also, two named presets can be parameter-identical on one side because some branches share an `or` (e.g. the SPX 2026-03-29 and 2026-04-05-parity presets share many LOW values).

### 4.2 The optimizer research narrative

The presets are snapshots from successive optimizer campaigns across three instruments (Gold 1D, DAX 1M, SPX 1D), each tagged with trial number, raw score, and (for later runs) a deflated score. The provenance vocabulary in the header (lines 4–58) is precise:

- **Path A** (lines 26–34): the earlier fold-based scorer family. The 5k run uses "14 bar-based folds"; the 2026-05-18 run uses "5 legacy folds."
- **Scorer v3** (line 47): **Hungarian matching + bootstrap-CI LCB + stable reference scale N=50.** Hungarian matching optimally pairs predicted turns to ground-truth turns (assignment problem); bootstrap-CI LCB scores by the *lower confidence bound* of a bootstrap distribution (rewards stability, penalizes lucky variance); the fixed N=50 makes scores comparable across runs.
- **Scorer v4** (line 58): **structural-nest [50, 100, 200] oracle**, validated against a hand-curated oracle of 7 famous SPX bottoms — Run 3's v4 pass "catches 7/7 famous SPX lows."
- **Raw vs deflated score**: later runs report both, e.g. "score 0.0679 (deflated 0.0667)" (line 41). The deflated value is a haircut for multiple-comparison / selection bias — picking the best of many trials inflates the apparent score, so the deflated figure is the more honest estimate.
- **IS/OOS degradation** is reported candidly per side (see trust map below).
- **Edge voting V14 → V15** is the headline algorithmic change (§5.4): a vote fires only on a fresh false→true transition, killing burst-cluster false tops. It is toggled *asymmetrically per side and per preset*; only the six 2026-05 SPX presets engage it at all.

### 4.3 The 14 presets, grouped

1. **Gold 1D** — `Gold 1D Current` (the production default): "exact final 500-run HIGH/LOW winners" (line 5), HIGH trial 292 / LOW trial 465. The polished flagship.
2. **DAX 1M** — `DAX 1M Latest`: a staging "paste target" for the latest DAX 1-minute snapshot (lines 6, 14–16), HIGH trial 226 / LOW trial 292, scores ~2.48 / ~2.87 (different scorer scale).
3. **SPX 1D family** (the bulk):
   - `2026-03-29 18:35 Trial 1225` — diagnostic LOW-only (score 0.0), HIGH left on Gold config for comparison.
   - `2026-04-05 Parity High+Low` — diagnostic testing both sides at once (HIGH 4837, LOW 1225).
   - `2026-04-05 Heuristic Structural` — **NOT optimizer-derived**; manually tightened from the parity pair.
   - `2026-04-05 20:59 Salvaged Best` — best recovered from a **corrupted overnight run** (HIGH 4101, LOW 2516).
   - `2026-05-18 21:51 Path A` — Path A on 5 legacy folds; **both sides flagged "likely outlier — params may be fragile."**
   - `2026-05-23 Path A 5k` and `Path A 5k Edge` — the *same* 5k winners (HIGH 425 / LOW 305, partial early-stopped run); the "Edge" variant is identical params with edge voting on.
   - `2026-05-23 V15 Run 1` and `Run 1 Selective` — V15 runs (Selective tightens gating, e.g. `min_agreement_low` 0.793 vs 0.128).
   - `2026-05-23 V15 Run 2` — Scorer v3 winners (HIGH 229 caution / LOW 130 clean).
   - `2026-05-24 V15 Run 3` — Scorer v4 winners (HIGH 167 noise-floor / LOW 169 v4 keeper).
4. **Legacy V11** — `Legacy V11 Gold`: the original V11 config, preserved as the ternary fallthrough default.

### 4.4 Trust map (per the author's own header)

| Verdict | Preset / side | Evidence |
|---|---|---|
| **TRUSTED** | Gold 1D Current | "final 500-run winners" (line 5). |
| **TRUSTED** | Run 2 LOW (trial 130) | IS=0.189 ≈ OOS=0.194 — "generalizes cleanly to modern regime" (line 46). |
| **TRUSTED (best)** | **Run 3 LOW (trial 169)** | IS=0.029 → OOS=0.089 (**3× better OOS**), stability 99.7%, "the **v4 keeper**" (lines 56–57). |
| **CAUTION** | Run 2 HIGH (trial 229) | IS=0.065 → OOS=0.032 (**~50% degradation** on 2004–2026, line 43). |
| **NOT SIGNAL** | Run 3 HIGH (trial 167) | IS=0.025 → OOS=0.000. "HIGH is the **noise floor** on SPX 1D… Keep on chart for comparison, not signal" (lines 52–53). |
| **FRAGILE** | 2026-05-18 Path A (both sides) | Stability probe flagged both "likely outlier — params may be fragile" (line 34). |
| **DIAGNOSTIC** | Trial 1225, Parity, Salvaged Best | Test/recovery configs, not deployment. Salvaged Best came from a corrupted run — treat with suspicion despite no explicit caution. |
| **HAND-TUNED** | Heuristic Structural | Not optimizer-validated. |
| **PROVISIONAL** | DAX 1M Latest | "Staging / paste target"; no IS/OOS numbers given. |

> **Scores are NOT comparable across presets.** DAX scores (~2.5–2.9) and SPX scores (0.0–0.33) live on different scorer scales/difficulties. A 0.000000 score (Trial 1225 LOW) means "diagnostic on a hard scale," not "broken." The author deliberately keeps bad presets on the chart for visual comparison — an included preset is not necessarily a recommended signal source.

---

## 5. I/O & the Pine↔Python Parity System

### 5.1 Inputs and declaration

`indicator("Speculatores V15 Presets - Per-Side Regimes (edge voting)", overlay=true, max_bars_back=5000)` (line 2). The large `max_bars_back=5000` is essential — the scale loop and PIR/ER history scans reach hundreds of bars back (`scale_end` up to 354), and Pine's auto-inference would otherwise under-allocate and throw a "bar index too far" error.

Only **4 user inputs** exist (lines 61–66): one 14-option `preset` dropdown (default "Gold 1D Current") that resolves the entire HIGH+LOW parameter vector, plus three cosmetic display toggles — `show_bg`, `show_baseline`, `show_panel` — that gate visuals only and never affect signal computation.

### 5.2 The parity shim (the central correctness mechanism) — lines 73–113

**The bug it fixes.** Pine's stateful series built-ins (`ta.sma`, `ta.lowest`, `ta.highest`, `ta.atr`, `ta.stdev`, …) are *not* pure functions of their arguments: each *written call site* owns ONE internal history buffer that Pine advances once per bar, assuming a stable window length. Placing such a built-in inside a loop with a **variable-length argument** (`for s = scale_start to scale_end … ta.sma(close, s)`) makes every iteration stomp the previous scale's accumulated state. The comment (lines 73–85) records the empirical damage: a **0.14-mean / 0.9-max divergence vs the Python reference and a 2× signal-count gap** on the Path A 5k preset.

**The fix — stateless SMA via `ta.cum`.** One module-scope cumulative sum `_csum_close = ta.cum(close)` (line 87, evaluated at exactly one call-site once per bar). Then the SMA of length `s` ending at offset `back` is a constant-time difference (lines 90–93):

```pine
sma_at(s, back) => (_csum_close[back] − _csum_close[back + s]) / s
```

`pir_for_scale(s, lb)` (lines 98–113) rebuilds the `close/SMA_s` ratio range bar-by-bar with `math.min`/`math.max` accumulators and pure `close[back]` history access — **no per-call-site state to corrupt across scales.** This makes the agreement loop a deterministic pure function of price history, which is exactly what a Python reimplementation can reproduce.

**Scope of the discipline.** The rule is *"never call a stateful `ta.*` inside the variable-`s` loop,"* NOT "never use them." Stateful `ta.*` are still used legitimately at **fixed single call-sites** (trend SMAs 403/409, `vol_surge` 439, `vola_raw` 456, price gates 465–466, baseline pivots 491–492) and in explicit unrolled fixed-length flag chains (`low_scale_flag`/`high_scale_flag`, lines 306–387), where one-buffer-per-site is the correct, intended behavior.

### 5.3 Visuals and info panel

- **Background** (594–600): `bgcolor` red/green at ~78% transparency on a fired pivot, gated by `show_bg`.
- **Markers** (602–603): `plotshape` red `triangledown` above / green `triangleup` below, on the rising edge `pivot_x and not pivot_x[1]`.
- **Baseline pivots** (604–605): purple diamonds at confirmed `ta.pivothigh/low`, drawn `offset=−20` so the diamond sits on the true pivot bar (20 bars back) rather than the confirmation bar.
- **Info panel** (607–626): a `2×9` bottom-right table mirroring internals — preset name, HIGH/LOW `S/C/PG` config, vote counts `X/max`, accel, drift, `Agreement H|L`, and the current signal. A human-readable twin of the numeric exports below.

### 5.4 V15 edge voting — `edge_or_state` (lines 506–521)

```pine
edge_or_state(v, win, use_edge) =>
    prev = bar_index >= win ? v[win] : false   // 520: Pine v6 has no nz/na for series-bool
    use_edge ? (v and not prev) : v             // 521
```

- **State vote (V14 default, `use_edge=false`):** returns `v` as-is — "is the condition true *now*."
- **Edge vote (V15, `use_edge=true`):** returns `v and not v[win]` — true only on a false→true transition within the `edge_window`. This "spends" a vote once so it won't keep re-firing while the indicator sits true through a long rally (the burst-cluster false-top fix, lines 506–511).

All eight votes per side are wrapped uniformly (lines 530–546). When the toggle is false the `*_eff` bools are byte-identical to V14 state bools, so presets are backward-compatible.

### 5.5 The `dbg_*` exports — the Python-parity diff surface (lines 628–658)

~30 `plot(..., display=display.data_window)` calls that **do not draw on the chart** — they surface a named numeric value per bar in TradingView's Data Window, *solely* to be diffed bar-by-bar against the Python reference. Booleans are exported as `1.0/0.0` and integer counters as `*1.0` floats so every column is directly subtractable. The header (line 628) labels them "high-side parity focus" — the set is deliberately weighted toward HIGH, the case the shim fixed. Granular per-vote exports `dbg_high_vote_*` (lines 646–655) exist so that, when a divergence appears, the diff pinpoints exactly *which* vote (and underlying indicator) drifted — the affordance that surfaced the stateful-buffer corruption in the first place.

They map onto the columns of `data/enriched/SPX_1D_18710201_20260318_TV_V11.csv` (e.g. `agreement_high`, `scales_high/low`, `n_scales`, `scale_div_high`, `dur_high`, `ph_confirms`, `gate_pass_high`, `cd_high`, `price_gate_high`, `er_val`, `baseline_pivot_high`). Full column mapping in §6.3.

> **Export-naming gotcha:** matching must be by *meaning*, not a uniform `dbg_` prefix — some columns are plain-named (`agreement_high_side`, `momentum_velocity_high`, `ph_confirms`, `er_val_high`). `agreement_high_high` is even exported twice under two names (lines 629, 630) — duplicate value, not two series.

### 5.6 Alerts and version drift

Two `alertcondition`s fire on the rising signal edge (lines 661–662) — same predicate as the plotshapes. **Version drift is real:** the filename and title say **V15**, but the alert message strings say *"Speculatores **V14**: pivot HIGH/LOW"* and the comment at line 507 calls STATE evaluation the "V14 default." This is **stale copy, not a behavioral difference** — with `use_edge_voting=false` every `*_eff` vote collapses to V14 state behavior, and the V15 edge layer is opt-in per preset. Anyone keying automation off the alert text must not assume it reflects the file version.

---

## 6. Data Corpus

Three tiers under `C:\Users\kuben\Desktop\Projekte\cfd10\data\`. All timestamps are **unix epoch seconds, UTC**, and can be **negative** for pre-1970 reconstructed daily history (the 1871 S&P composite). The README schema is `time,open,high,low,close,volume`.

> **Environment note:** Python is NOT installed on this machine (`python3`/`python` hit the Windows Store alias, exit 49). All timestamp/CSV verification was done via PowerShell `[DateTime]` math.

### 6.1 `data/raw` — 14 hand-picked CSVs

The original curated set: indices (SPX, NDX, DAX, VIX), commodity futures (GC1, SI1, WTI), and FX (EURUSD). **12 of 14 conform** to the lowercase `volume` schema. Two non-conformers:

- **`SPX_1D_18710201_20260318.csv`** is NOT 6-column OHLCV — it is the **41-column enriched feature file mislocated in `raw/`**, byte-identical to the `enriched/` copy *except the final column is capital `Volume`*. The enriched copy has **one extra data row** (25191 vs 25190). README line 62 documents it as a provenance copy of the March-20 workspace dataset.
- **`DAX_1D_19700102_20260324.csv`** is 6-column but also uses capital `Volume`.

| File | Data rows | Cols | Coverage (UTC) |
|---|---|---|---|
| SPX_1D_18710201_20260318 *(enriched, in raw/)* | 25190 | **41** | 1871-02-01 → 2026-03-18 |
| SPX_1D_20170428_20260318 *(clean slice)* | 2234 | 6 | 2017-04-28 → 2026-03-18 |
| DAX_1D_19700102_20260324 | 14149 | 6 (`Volume`) | 1970-01-02 → 2026-03-24 |
| VIX_1M_20241104_20260313 | 236155 | 6 | 2024-11-04 → 2026-03-13 |
| SPX_1M / NDX_1M / DAX_1M (long + short pairs) | ~21k–145k | 6 | Jan/Feb 2025 → Mar 2026 |
| GC1_1M / SI1_1M / WTI_1M / EURUSD_1M | ~24k–27k | 6 | single Feb–Mar 2026 window |

Layout patterns: SPX/NDX/DAX 1M each come as a **long base** file (~Jan/Feb 2025 → Feb 2026) plus a **short rolling top-up** (~Dec 2025 → Mar 2026) that overlaps and extends ~2 weeks past the long file. Commodities/FX exist only as a single short 1M window.

**Volume = 0 is era-dependent, not blanket-per-index.** VIX_1M is 100% zero (a calculated index has no volume). DAX_1D is *partially* zero — 8502/14149 early (1970s–80s) rows are 0, later bars carry real volume. SPX_1D recent carries non-zero synthetic index volume (~2.2–3.0e9). Isolated single-bar zero-volume prints exist inside otherwise-real 1M files (2 in SPX_1M long, 4 in DAX_1M long) — thin/auction prints with valid OHLC.

**VIX is purely exogenous.** A grep of all 662 Pine lines for `request.security`, `security(`, `syminfo`, `ticker(`, `barmerge`, and `vix`/`pos_vix` returned **zero matches**. The script's only external-data primitive is `ta.cum(close)` plus OHLCV built-ins. VIX percentile (the enriched `pos_vix` column) is computed entirely offline in the Python enrichment layer and is **never read by Pine**. The on-chart `vola_*` inputs are the script's own realized-volatility proxies (ATR/StdDev/Intraday/GJR/HAR), not VIX.

### 6.2 `data/raw_v16` — 80 CSVs, the V16-era bulk export (~51.2 MB)

TradingView's verbatim chart-export naming: **`{EXCHANGE}_{TICKER}, {minutes}_{hash}.csv`** (note the literal `, ` comma+space, the `!` on futures, the `.` on class shares). Parse with a regex like `^(.+?),\s*([0-9]+D?)_([0-9a-f]+)\.csv$`, never naive splits. The hash is a non-semantic per-download id. Schema is identical to `data/raw` **except the volume column is capital `Volume`** (all 80 files).

**Timeframe token = minutes-per-bar:** `1`=1min, `60`=1H, `240`=4H; `1D`=daily (literal, not minutes — so don't `int()`-parse the token).

| TF | Files | Composition |
|---|---|---|
| `1` (1min) | 12 | 11 core instruments + 1 SI1! duplicate |
| `60` (1H) | 11 | the 11 core instruments |
| `240` (4H) | 11 | the 11 core instruments |
| `1D` (daily) | 46 | 28 single stocks (BATS_, 1D-only) + 7 FX_IDC pairs (1D-only) + 11 core minus WTI & OANDA-EURUSD (no 1D here) |
| **Total** | **80** | |

**Universe categories:** indices/ETF-proxies (5: SPX, NDX, DAX, plus VT & VWCE world-equity ETFs); commodity futures (4 continuous `!`: GC1!, SI1!, PL1!, PA1! + WTI via BlackBull CFD); FX (8: 7 FX_IDC daily + OANDA_EURUSD intraday); single stocks (28 BATS_ mega-caps, daily only). **EURUSD is split across providers** — OANDA for intraday, FX_IDC for daily — so assembling full multi-TF EURUSD requires joining two feeds.

**Coverage pattern:** 1min files are uniformly short (~20–25k bars ≈ recent ~2.5 months — TradingView caps intraday export); 60/240 reach back years (VT 1H to 2014, 4H to 2008); 1D reaches instrument inception (SPX 1871, DAX 1970, USDRUB 1994; PLTR 2020 and VWCE 2019 naturally short). All series end ~2026-05-27/28, dating this export to late May 2026. `SP_SPX, 1D` starts at `time=-3121407238` (1871-02-01, OHLC=4.5, volume=0) — the **same reconstructed S&P composite** as the enriched file, confirming shared lineage.

**The SI1! duplicate — confirmed.** `COMEX_DL_SI1!, 1_6bcd3.csv` (25005 rows) and `…1_8d38f.csv` (25003 rows) are **byte-identical for the first 25004 lines**; `6bcd3` is a strict 2-bar superset (ends 20:18 vs 20:16 UTC, 2026-05-28), exported ~2 minutes apart in the same session. **Not different windows — keep `6bcd3`, drop `8d38f`.** This is the only true duplicate in the directory; double-loading would double-count the silver 1min series.

### 6.3 `data/enriched` — the Python ground-truth feature dump

One file: **`SPX_1D_18710201_20260318_TV_V11.csv`** — **41 columns** (the inventory's "42" is an off-by-one miscount), **25191 data rows** (= 25192 lines incl. header). "TV_V11" decodes as **TradingView-native export shape, Python feature-pipeline version 11.**

**It is an OLDER parity snapshot.** The Pine file is *v15* but the data is *V11* — the data artifact predates the script, so it is a parity/regression snapshot of an earlier engine, not the output of any current v15 preset. Two hard tells:

1. **`n_scales` is a constant 299 for all 25191 rows.** Every v15 preset derives `n_scales` from `(scale_end−scale_start)/scale_step + 1` (Gold = 90 high / 16 low; Legacy = 64) — none produce 299. V11 used a *dense ~299-contiguous-scale sweep* (`scales_high` max 287, `scales_low` max 298 are consistent), later replaced by the sparse stepped grid.
2. **`pos_vix` is 100% empty.** No v15 preset has a VIX input, so the column is reserved/forward-looking — a Python feature the current Pine family cannot populate.

**Signal tiers — Strong vs Regular (mutually exclusive).** The Python pipeline splits each side's single Pine boolean into two tiers by vote saturation:

- **Strong High/Low** = full consensus: every Strong row has `confirms == 3` (9 High rows, 62 Low rows; all `gate_pass=1`, `cd=0`).
- **Regular High/Low** = partial vote: `confirms ∈ {1,2}` (1050 High, 448 Low rows).
- No bar is both (Strong is NOT a flagged subset of Regular — treating it as "a stronger Regular" double-counts).

**Row-count reconciliation:** enriched 25191 data rows = raw full-SPX exactly (1:1 feature augmentation). v16 `SP_SPX, 1D` = 25240 (+49 bars), attributable to a later capture end-date plus SP-feed session/holiday handling vs the Python pipeline.

**Full 41-column → Pine mapping:**

| # | Column | Pine counterpart | Meaning |
|---|---|---|---|
| 1–5 | time, open, high, low, close | builtins; `close` feeds `_csum_close` (87) | OHLC; unix-sec UTC time. |
| 6–7 | Strong High / Strong Low | `pivot_high/low` at `confirms ≥ 3` (585–586) | High-conviction tier. |
| 8–9 | Regular High / Regular Low | `pivot_high/low` at `confirms ∈ {1,2}` | Lower-conviction tier. |
| 10–11 | agreement_high / agreement_low | `agreement_*_*` / `dbg_agreement_*` (303–304, 630) | `scales_*/n_scales`. |
| 12–13 | scales_high / scales_low | `scales_*_*` / `dbg_scales_*` (632–633) | Extreme-scale counts (max 287 / 298). |
| 14 | n_scales | `n_scales_*` / `dbg_n_scales_high` (634) | **Fixed 299 in V11** vs preset-derived in v15. |
| 15 | pos_detect | `pir_detect_high` (394) | Detect-scale PIR ∈ [0,1]; LOW uses `1 − pos_detect`. |
| 16–17 | scale_div_high / scale_div_low | `scale_div_*` (397–398) | `pir_detect − agreement` (HIGH); `(1−pir_detect) − agreement` (LOW). Verified at row 15000: `(1−0.5175)−0.6120 = −0.1295`. |
| 18–20 | slope_sma / linreg_slope / trend_strength | `slope_val` / `linreg_slope` / composite (404–407) | Normalized trend estimators. |
| 21–22 | dur_high / dur_low | `dur_at_*` (416–430) | Extreme-dwell counters. |
| 23 | vol_surge | `vol_surge_*` / `dbg_vol_surge_high` (439–440, 648) | Fast/slow volume ratio. |
| 24 | mom_diverge | `mom_diverge_*` (445, 451) | `price_ret × vol_ret`. |
| 25–27 | vola_raw / vola_pos / vola_slope | `vola_raw/pos_*` (456–461); `vola_slope` no dbg export | Volatility series (`vola_raw` empty in 3 earliest rows). |
| 28–29 | ph_confirms / pl_confirms | `ph/pl_confirms` (550–568, 641) | Vote tallies (0–3 here). |
| 30–31 | gate_pass_high / gate_pass_low | `gate_pass_*` / `dbg_gate_pass_high` (576–577, 643) | All hard gates passed. |
| 32–33 | cd_high / cd_low | `bars_since_*_signal` (580–581, 644) | Cooldown counters. |
| 34–35 | price_gate_high / price_gate_low | `price_*_ok` (465–466, 656) | New-high/low gate. |
| 36–37 | er_val / er_gate | `er_val_*` / `er_gate_ok_*` (472–480, 657) | Kaufman ER + falling-edge gate (`er_gate=1` in 12057/25191 rows). |
| 38–39 | baseline_pivot_high / low | `baseline_ph/pl` (491–492, 658) | Sparse confirmed pivots (346 / 399 non-empty). |
| 40 | pos_vix | **NO Pine counterpart** | VIX PIR — Python-only, 100% empty. |
| 41 | volume | builtin | Feeds vol_surge & mom_diverge. |

> **Preset-lineage gotcha:** the Gold 1D Current HIGH side enables only ONE vote (`use_momentum_velocity_high`), giving `max_votes_high=1` — yet the CSV shows `ph_confirms` up to 3. The artifact was generated under a *richer V11/Legacy-style* vote configuration. Match it against the **Legacy V11 Gold lineage**, not Gold 1D Current — do not expect bit-exact parity between this CSV and any single v15 preset.

---

## 7. How It All Fits Together

The research loop closes a circle between TradingView (Pine, on-chart) and Python (optimizer + reference, offline):

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  (1) TradingView export                                                 │
   │      data/raw  (14 curated CSVs)                                        │
   │      data/raw_v16 (80 multi-asset × multi-TF CSVs)  ── OHLCV, UTC ──┐   │
   └────────────────────────────────────────────────────────────────────┼───┘
                                                                          ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │  (2) Python optimizer                                                  │
   │      Optuna-style trials over folds → Scorer (Path A → v3 → v4)        │
   │      deflated scores, IS/OOS holdout, stability probe                  │
   │      separate searches for the best HIGH-side and best LOW-side params │
   └────────────────────────────┬─────────────────────────────────────────┘
                                 ▼  winning HIGH + LOW parameter vectors
   ┌──────────────────────────────────────────────────────────────────────┐
   │  (3) Pasted as a Pine preset                                           │
   │      a new is_<name> branch in every per-side ternary (lines 191–285)  │
   │      tagged with trial #, raw/deflated score, IS/OOS verdict in header │
   └────────────────────────────┬─────────────────────────────────────────┘
                                 ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │  (4) Pine renders signals on chart                                     │
   │      pivot_high / pivot_low (585–586) → red/green triangles + alerts   │
   └────────────────────────────┬─────────────────────────────────────────┘
                                 ▼  ~30 dbg_* data-window exports (628–658)
   ┌──────────────────────────────────────────────────────────────────────┐
   │  (5) Parity proof                                                      │
   │      dbg_* values  ──diff bar-by-bar──▶  Python reference table        │
   │                       data/enriched/SPX_1D_..._TV_V11.csv              │
   │      divergence localized to the exact vote via dbg_high_vote_*        │
   └──────────────────────────────────────────────────────────────────────┘
```

**The role of each tier:**

- **`data/raw` and `data/raw_v16` are the optimizer's fuel.** `raw` supported single-asset runs; the V16 bulk export (47 instruments × up to 4 timeframes) is what enables the cross-asset / cross-timeframe "multi-asset presets." Both share the same OHLCV contract — the only friction is the `Volume` vs `volume` header case, which any shared loader must normalize.
- **The Python optimizer is where the science happens.** It runs the trials, applies the scorers (Hungarian matching + bootstrap-CI LCB for v3, structural-nest oracle for v4), deflates scores for selection bias, and reports IS→OOS degradation and stability-probe verdicts. Crucially it optimizes the HIGH and LOW detectors *independently*.
- **A preset is the optimizer's frozen output, pasted into Pine.** The header annotations (trial #, deflated score, "v4 keeper" / "noise floor" / "likely outlier") are the audit trail of that paste.
- **Pine is the deployment surface** — it renders the chart signals a human trades from, and fires alerts.
- **`data/enriched` is where parity is proven.** It is the Python side's **V11 ground-truth feature snapshot** for SPX daily 1871–2026: the same engine, computed offline, column-for-column. The `dbg_*` exports are the Pine side of that same per-bar table. Diffing them is how the stateful-buffer corruption was discovered (0.14-mean / 0.9-max divergence, 2× signal gap) and how the parity shim was validated. Because the enriched file is *V11* while the Pine is *v15*, it is a **regression snapshot of an earlier engine generation** (fixed 299-scale sweep, empty `pos_vix`, Legacy-lineage vote config) — useful for proving the parity *methodology*, but not bit-exact against any current v15 preset.

In short: **the optimizer finds parameters offline, Pine renders them on-chart, and the enriched reference + `dbg_*` exports keep the two implementations numerically honest.**

---

## 8. Observations & Open Questions

### Cross-cutting gotchas (high-value, easy to miss)

- **The parity shim is scoped to the loop only.** Stateful `ta.*` remain correct (and used) at fixed single call-sites and in the unrolled SPX flag chains (306–387). Refactoring those flag chains *into a loop* would silently reintroduce the exact bug the shim was built to fix.
- **Two deliberate HIGH/LOW asymmetries** (§2.6): the extra pivot-drift veto on HIGH only, and the opposite-sign confirm-bias where *both* sides key off "drift up" but move the required count in opposite directions. A naive symmetric reading is wrong.
- **`scale_div` is sign-asymmetric:** HIGH = `pir_detect − agreement`, LOW = `(1 − pir_detect) − agreement` (397–398). A symmetric Python port would mis-sign the LOW divergence.
- **Volume vote reciprocal:** a HIGH vote means volume is *drying up* (`< 1/thresh`, line 524), not surging — easy to misread.
- **GJR sign vs HAR sign:** GJR's *sign* discriminates top vs bottom (484–485); HAR and volatility-elevated use the *same* direction on both sides — only the threshold differs.
- **The ER gate cannot pass on the first bar:** `er_val < nz(er_val[1], 0.0)` effectively requires `er_val < 0` with zero history.
- **Edge-voting warm-up:** on the first `win` bars `prev` is forced false (line 520, Pine v6 lacks `nz`/`na` for series-bool); a Python reference must replicate this or early-history parity diverges.
- **pivot-drift is excluded from `max_votes`** (570–571) yet contributes to `confirms` (552/562); `required` is min-clamped to `max_votes`, decoupling the two, so an over-large `confirm_count` silently saturates.
- **Version/label traps:** filename says V15, alerts say V14 (stale copy); the inventory's "42 columns" is actually 41; the enriched-in-`raw/` file is enriched (41-col), not OHLCV; `Volume` vs `volume` header case differs across the three data tiers.
- **Negative epochs:** the 1871–1969 reconstructed history uses negative unix-seconds; unsigned/millisecond parsers will misread the first several thousand daily rows.
- **Strong ⊄ Regular:** the two enriched signal tiers are mutually exclusive, not nested.

### Risks

- **Several flagged presets remain on-chart as signals.** Run 3 HIGH ("noise floor"), Run 2 HIGH (~50% OOS degradation), and 2026-05-18 Path A (both "likely outlier") are kept for comparison — selecting them as live signal sources would be a mistake. The author's per-side CAUTION notes are the guardrail.
- **Salvaged Best** (0.331818 HIGH) came from a *corrupted* overnight run; its provenance is unverified despite carrying no explicit caution.
- **DAX 1M Latest is provisional staging** with no IS/OOS validation.
- **Parity is proven against a V11 snapshot, not a v15 preset.** There is no in-repo evidence that any *current* v15 preset has been diffed against a fresh v15-generation enriched reference.
- **The SI1! duplicate** will double-count silver 1min if both files are globbed into the optimizer.

### Open questions worth surfacing

1. **The Python optimizer/reference source was not located in this repo.** Confirming the exact column↔export alignment, the Regular→Strong promotion rule (hard `confirms ≥ 3`? `≥ max_votes`? a fraction?), and the V11 dense-scale sweep parameters (is it `s = 1..299`?) all require that external code.
2. **Gold HIGH `pivot_drift_confirm_bias_high`** (line 230) has no explicit `is_gold_current` branch and falls through to the trailing `:1` — so Gold HIGH gets a +1 bias when drift is up. Is this intended (the LOW bias is explicitly set to 0 on line 269) or an oversight?
3. **Active-preset toggle values for Gold 1D Current** (`use_trend_high`, `use_volume_high`, gate toggles, `momentum_velocity_mode_*`, `vola_method_*`) were not fully read; they determine which votes/gates are actually live and the effective `max_votes`/`required`. Note the §6.3 finding that Gold HIGH appears to enable only one vote.
4. **Are the hand-unrolled SPX scale ladders** (306–387, using *stateful* `ta.sma` via `low_scale_flag`/`high_scale_flag`) parity-maintained, or accepted as approximate for those presets?
5. **Dead parameters:** are `vola_low_pct` and `vola_slope_lb` consumed by a code path outside lines 68–547, or are they vestigial from V11?
6. **CSV columns without a 1:1 named export** (`pos_detect`/`slope_sma`/`linreg_slope`/`trend_strength`/`vola_slope`/`pos_vix`) — is parity checked on these via a different export set or not at all?
7. **Data provenance:** is there meant to be a clean 6-col SPX full-history daily export (the long series exists only as the enriched 41-col file)? Is DAX's zero-volume early segment authentic or backfilled? Which SPX row count (25190 raw vs 25191 enriched) is authoritative, and does the extra row sit at head or tail?
8. **Multi-asset data gaps:** where does the pipeline source daily WTI and intraday-only OANDA-EURUSD (no 1D in `raw_v16`)? Does the optimizer reference files by raw TradingView filename (hash-brittle) or a normalized symbol key? Is the ~2.5-month 1min window sufficient, or is longer 1min history expected to be stitched from `data/raw`?
9. **Partial 5k run:** the Path A 5k winners come from an early-stopped run (~341/225 trials per side) — was it ever resumed, and would the final winners differ?
