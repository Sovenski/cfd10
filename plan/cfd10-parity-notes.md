# Python ↔ Pine Parity Ledger

Living record of every deliberate deviation between the Python feature port and
the Pine source (`pine/speculatores_v15_presets_gold.pine`). Each must be either
(a) proven immaterial to **signal** parity, or (b) reconciled before the Phase 8
bar-by-bar parity verification. "Same signals on the same data" is the contract;
warm-up/quiet-region numeric differences only matter if they can flip a signal.

| # | Feature / file | Pine behaviour | Python behaviour | Impact on signal parity | Reconciliation plan (Phase 8) |
|---|----------------|----------------|------------------|-------------------------|-------------------------------|
| 1 | `sma_pir.py` csum boundary | `na` when `back+s > i` at series start | index `-1` treated as `0.0` → one extra valid bar at the very start | None — first bar can't pass gates (no history) | Add strict-`na` mode flag; verify the leading bar isn't a signal in the Pine export |
| 2 | `trend.py`, `momentum.py` warm-up | `nz(...)` makes insufficient-history → `0.0` | warm-up → `np.nan` | None — warm-up bars are dropped from train/eval and never fire | Apply `nz`→0.0 in the bank's "Pine-parity" assembly mode for the bars the student actually evaluates |
| 3 | `efficiency.py` flat window | `er_path==0` → `er_val=0.0` (NOT na) | preserved: `0.0`, distinct from warm-up NaN | None (already faithful) | Keep — downstream gate `er_val < nz(er_val[1],0)` depends on the 0.0 |
| 4 | `trend.py` linreg | `ta.linreg(src,len,off)` single OLS line at offset; `linreg(.,L,0)-linreg(.,L,1)` = OLS slope | closed-form OLS slope (== numpy.polyfit to 1e-9) | Interpretation risk — affects trend vote | **Verify against Pine export** (this is a genuine unknown) |
| 5 | `vola.py` ATR | `ta.atr` = Wilder RMA of true range | Wilder RMA (SMA-seeded) | Low — affects vola vote magnitude | Verify RMA seeding vs export |
| 6 | `vola.py` stdev | `ta.stdev` default `biased=true` (population) | population (÷N) | Low | Verify vs export |
| 7 | `pivots.py` alignment | `ta.pivothigh` confirms `lb` bars late; value plotted at offset `-lb` | mask flags bar `i` at its own index; lb-bar latency only as warm-up False band | Medium — pivot-drift uses confirmed-pivot history; alignment matters | Shift to confirmation-bar index `i+lb` in assembly if the export shows it; add test |
| 8 | `momentum.py` mom_divergence | `price_ret * vol_ret`, `vol_ret=(vol-vol[L])/max(vol[L],1)` | faithful; **unbounded** (~±7.7e7 on SPX 1D — volume 0→billions) | None for trees (scale-invariant splits); **bad for the NN** | Signed-log / winsorize for the teacher's input only; trees & Pine see raw |

## Standing conventions (apply project-wide)
- Feature layer emits **NaN** in warm-up; the bank exposes a `pine_parity` assembly mode that re-applies `nz`→0.0 exactly where Pine does, used only when generating signals to diff against a Pine export.
- All "true TradingView parity" claims are **provisional** until checked bar-by-bar against the user's `dbg_*` export (Phase 8). The scalar-vs-vectorized 1e-9 tests only prove internal self-consistency.
- Items #4 and #7 are the **highest parity risk** and get dedicated attention in Phase 8.
