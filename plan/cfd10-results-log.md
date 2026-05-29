# cfd10 Results Log

Honest, chronological record of out-of-sample experiments. Event precision/recall/F1
at 3-bar tolerance, purged + embargoed walk-forward CV, thresholds tuned on the train
fold only. **No Pine-signal ("Run 3") comparison yet** — that needs the user's export
(Phase 8). All numbers are oracle-vs-detector.

## E1 — GBDT baseline, single-asset SPX 1D, LOOSE oracle (the yardstick)
- Oracle: nest=(5,10,20,40), drawdown 5%, horizon 40, tau 0.5/0.25 → ~0.6–0.8% positives.
- **LOW: F1 0.249** (P 0.152, R 0.696), stable across 6 folds (0.19–0.34). ~20× lift over base rate.
- **HIGH: F1 0.106** (P 0.062, R 0.358). ~10× lift, weaker but not zero.
- Bug fixed here: oracle weight col is 0 on non-turns → zeroed the negative class; floored to `1 + score` on positives.
- Takeaway: frequent, smaller swings ARE learnable from the 36-feature bank; LOW > HIGH.

## E2 — GBDT, multi-asset POOLED (46 daily assets, 467k rows), STRUCTURAL oracle
- Oracle: nest=(20,50,100,200), drawdown 10%, horizon 60, tau 0.6/0.3 → 0.22% positives (the user's "big-n structural" preference; famous-lows-validated).
- Fix required: per-side class balancing `scale_pos_weight = n_neg/n_pos` — without it the 0.22% rate made the model predict nothing (F1=0 collapse).
- **LOW: pooled F1 0.126** (P 0.092, R 0.20); **SPX-subset F1 0.113**.
- **HIGH: F1 ≈ 0.000** (tp=0) — structural tops not separable by the GBDT.
- Label volume: structural oracle gives **738 LOW / 1034 HIGH positives pooled** vs **47 / 33 on SPX alone**.

### Interpretation (important, and partly confounded)
- **Pooling works for its purpose:** it lifts the structural oracle from *untrainable* (33–47 SPX positives) to a real signal (SPX-LOW OOS 0.11). That is the clean, causal claim (fixed oracle, SPX-only would be degenerate).
- The "−0.136 vs E1" is **NOT** "pooling hurts" — E1 and E2 differ in BOTH oracle granularity and data. The honest decomposition: *structural* turns (rare, big) are intrinsically harder to predict than *frequent swings* (E1's loose oracle), regardless of pooling.
- **HIGH stays at the noise floor**, and structural HIGH is worse than loose HIGH. Tops remain the hard/possibly-unlearnable side (matches the Run-3 prior).

### Open levers
1. **TCN teacher** (deep, temporal) — the designed lever to capture structural-turn dynamics the per-bar GBDT misses (E1 showed a train→OOS gap = headroom). Built; trains on Colab GPU.
2. **Oracle granularity is a product decision** — rare big structural turns (hard, ~0.11 LOW) vs including mid-size swings (easier, ~0.25 LOW). Currently set to the user's structural preference; revisit if the teacher can't lift structural HIGH off the floor.
3. A clean **like-for-like ablation** (same oracle, single vs pooled; and loose vs structural at fixed pooling) would fully de-confound E1/E2 — cheap, worth running before any paper claim.
