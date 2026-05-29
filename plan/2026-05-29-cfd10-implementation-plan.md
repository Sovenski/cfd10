# cfd10 — Learned-Combiner Turn Detector — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. Execution here is driven by **dynamic Workflows** (user directive).

**Goal:** Replace Speculatores' hand-tuned voting layer with a learned combiner — keep the features, learn a deep teacher, distill to a Pine-portable rule student that reproduces signals exactly.

**Architecture:** Config-driven Python pipeline: data → parity-faithful feature bank → weighted multi-scale label oracle → deep temporal teacher (+ GBDT yardstick) → distilled shallow-tree student → Pine export + bar-by-bar parity verification. Purged/embargoed walk-forward OOS throughout. Per-timeframe specialized rulesets.

**Tech Stack:** Python 3.12 (Colab-matched, via `uv`), NumPy/Numba, pandas/pyarrow, scikit-learn, LightGBM, PyTorch, Hydra/OmegaConf, Optuna, pytest/ruff/mypy. Compute: local CPU for everything except teacher training (Colab T4/L4). Repo: `github.com/Sovenski/cfd10` (public now → private at the end).

**Source of truth for parity:** `pine/speculatores_v15_presets_gold.pine` (663 lines) + `plan/cfd10-system-understanding.md` + this repo's spec `plan/2026-05-29-learned-combiner-turn-detector-design.md`.

**Legend:** 🟢 LOCAL (build+test here) · 🟡 COLAB-GPU (smoke-test local, train on Colab) · 🔴 USER-GATED (needs Pine exports from user).

---

## File structure (locked decomposition)

```
cfd10/
├── pyproject.toml                 # uv, python 3.12
├── requirements.txt               # Colab deps (torch excluded — use Colab's GPU build)
├── requirements-dev.txt           # local: cpu torch, pytest, ruff, mypy
├── .gitignore                     # data/, outputs/, .venv/, *.pt, .env, settings.json
├── README.md
├── notebooks/colab_cfd10.ipynb    # uploaded notebook (mount drive, clone, install, run)
├── conf/                          # Hydra dataclass-backed configs
│   ├── config.yaml  data/*.yaml  oracle/*.yaml  features/*.yaml  teacher/*.yaml  cv/*.yaml  distill/*.yaml
├── src/cfd10/
│   ├── utils/            seed.py  logging_conf.py  env_record.py
│   ├── data_module/      schema.py  loader.py  resample.py  __init__.py(registry)
│   ├── feature_module/   sma_pir.py  trend.py  vola.py  momentum.py  efficiency.py  garch_har.py  pivots.py  bank.py  __init__.py
│   ├── label_module/     oracle.py  __init__.py
│   ├── cv_module/        splits.py  __init__.py
│   ├── eval_module/      events.py  metrics.py  deflation.py  report.py  __init__.py
│   ├── teacher_module/   datasets.py  models/tcn.py  models/__init__.py(registry)  train.py  calibrate.py
│   ├── student_module/   distill.py  export_pine.py
│   └── parity_module/    verify.py  pine_export_contract.md
├── pipeline/             build_features.py  fit_baseline.py  fit_teacher.py  distill.py  export_pine.py  verify_parity.py
└── tests/                mirrors src/ ; tests/data/ holds tiny fixtures
```

Each `src` file ≤ ~400 lines, one responsibility, factory+registry per module, `@dataclass(frozen=True)` configs, type hints, `logging` not `print`, `__all__` in every `__init__.py`.

---

## Phase 0 — Repo & harness bootstrap 🟢

**Outcome:** Public repo exists, clones, installs, `pytest` runs green on a smoke test, Colab notebook skeleton present.

### Task 0.1: Create repo + local scaffold
- Create: `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `.gitignore`, `README.md`, `src/cfd10/__init__.py`, `tests/__init__.py`, `tests/test_smoke.py`.
- [ ] `gh repo create Sovenski/cfd10 --public --description "Learned-combiner market turn detector (distills to Pine)"`
- [ ] `uv init` style `pyproject.toml`, `requires-python = ">=3.12,<3.13"`; `uv venv --python 3.12`; `uv pip install -r requirements-dev.txt`.
- [ ] `.gitignore` MUST exclude `data/ outputs/ .venv/ *.pt *.ckpt .env settings.json __pycache__/ .hydra/`.
- [ ] `requirements.txt` (Colab): `numpy pandas pyarrow numba scikit-learn lightgbm hydra-core omegaconf optuna matplotlib tqdm scipy` (NO torch line — Colab provides GPU torch). `requirements-dev.txt`: includes `torch --index-url .../cpu`, `pytest ruff mypy`.
- [ ] `tests/test_smoke.py`: `def test_import(): import cfd10; assert cfd10.__version__`
- [ ] Run `pytest -q` → PASS. Commit `chore: scaffold cfd10 repo`. `git push -u origin main`.

### Task 0.2: utils (seed, logging, env record)
- Create `src/cfd10/utils/{seed.py,logging_conf.py,env_record.py}` + tests.
- [ ] `set_seed(seed:int=42)->None` (random, numpy, torch, PYTHONHASHSEED, cudnn deterministic) — per repo reproducibility rule. Test: two `set_seed(0)` → identical `np.random.rand(3)`.
- [ ] `get_logger(name)->Logger`; `record_env()->dict` (python/torch/cuda/gpu). Commit.

### Task 0.3: Colab notebook skeleton
- Create `notebooks/colab_cfd10.ipynb` with cells: (1) mount drive, (2) clone+pull `Sovenski/cfd10` to `/content/cfd10` + `pip install -r requirements.txt`, (3) Hydra config overrides (GDrive data path, target TF), (4–8) call `pipeline/*.py` stages, (9) scorecard + Pine export download. Commit.

---

## Phase 1 — Data layer 🟢 (uses local `data/raw_v16`)

**Outcome:** Any (instrument, timeframe) loads to a clean canonical DataFrame; known row counts match.

### Task 1.1: schema normalization
- Create `src/cfd10/data_module/schema.py`, `tests/data_module/test_schema.py`.
- Interface: `CANONICAL_COLS = ("time","open","high","low","close","volume")`; `normalize(df)->DataFrame` (rename `Volume`→`volume`, coerce float64 OHLCV, int64 `time`, sort by time, drop dup timestamps keep-last, assert monotonic).
- [ ] Test contract: input with `Volume` col → output has `volume`, lowercase, sorted, no dup timestamps. Input with negative `time` (-3121407238) preserved as int (NOT parsed as date here). Commit.

### Task 1.2: loader + raw_v16 name parsing + dedup
- Create `src/cfd10/data_module/loader.py`, tests.
- Interface: `parse_v16_name(fname)->(exchange,ticker,tf_minutes,hash)` (regex on `"{EXCH}_{TICKER}, {MIN}_{hash}.csv"`, tf in {1,60,240,"1D"}); `load_csv(path)->DataFrame` (→ `schema.normalize`); `load_pair(root, ticker, tf)->DataFrame`.
- [ ] Test contract (against real local files): `load_csv("data/raw_v16/SP_SPX, 1D_a20e0.csv")` → 25239 data rows (25240 lines − header); `parse_v16_name("COMEX_DL_SI1!, 60_af791.csv") == ("COMEX_DL","SI1!",60,"af791")`.
- [ ] `KNOWN_DUPLICATES = {"COMEX_DL_SI1!, 1_8d38f.csv"}` — loader skips it; document the 6bcd3⊃8d38f finding in a comment. Test: a directory-load excludes it. Commit.

### Task 1.3: resample (for cross-TF ablation, build now)
- Create `src/cfd10/data_module/resample.py`, tests.
- Interface: `resample_ohlcv(df, rule)->DataFrame` (OHLC agg correct: open=first, high=max, low=min, close=last, volume=sum). Test: 4×1h bars → 1×4h bar values correct. Commit.

---

## Phase 2 — Feature bank 🟢 (PARITY-CRITICAL — formulas inlined)

**Outcome:** Vectorized features that mirror Pine semantics exactly. Pine line refs are normative.

### Task 2.1: stateless SMA + PIR (the parity lynchpin) — Pine L87–113
- Create `src/cfd10/feature_module/sma_pir.py`, tests.
- **Inline implementation (port of the `ta.cum` shim — this exact logic is required):**
```python
import numpy as np
def csum_close(close: np.ndarray) -> np.ndarray:
    # Pine: _csum_close = ta.cum(close); inclusive prefix sum. csum[i] = sum(close[0..i]).
    return np.cumsum(close)

def sma_at(csum: np.ndarray, close_len: int, s: int, back: int, i: int) -> float:
    # Pine sma_at(s,back) at bar i = (csum[i-back] - csum[i-back-s]) / s ; na if OOB.
    a_idx, b_idx = i - back, i - back - s
    if b_idx < -1 or a_idx < 0:
        return np.nan
    a = csum[a_idx]
    b = csum[b_idx] if b_idx >= 0 else 0.0
    return (a - b) / s

def pir_for_scale(close: np.ndarray, csum: np.ndarray, s: int, lb: int, i: int) -> float:
    # Pine pir_for_scale: ratio = close/sma_s scanned over last lb bars; PIR of current vs [lo,hi].
    sma_now = sma_at(csum, len(close), s, 0, i)
    if not (sma_now > 0):
        return 0.5
    val_now = close[i] / sma_now
    lo = hi = val_now
    for back in range(1, lb):
        smab = sma_at(csum, len(close), s, back, i)
        if back <= i and smab is not None and smab > 0 and not np.isnan(smab):
            rb = close[i - back] / smab
            lo = min(lo, rb); hi = max(hi, rb)
    return (val_now - lo) / (hi - lo) if hi != lo else 0.5
```
- Provide a **vectorized/Numba** equivalent for full-series speed, but keep the scalar version as the parity reference the vectorized one is tested against.
- [ ] Test: scalar vs vectorized agree to 1e-9 on a random 500-bar series. Test: PIR ∈ [0,1]. Test: on a monotone-rising series, late-bar PIR ≈ 1. Commit.

### Task 2.2: multi-scale agreement — Pine L115–126
- `agreement(close, csum, scale_start, scale_end, scale_step, pct_extreme, i)->(scales_high,scales_low,n,agree_high,agree_low)`; `lb=max(s,20)`. Test: counts match a hand-rolled loop over `pir_for_scale`. Commit.

### Task 2.3: trend, vola, momentum, efficiency, GJR/HAR, pivots
Each in its own file, ported from the cited Pine lines, with a numeric test vs hand-computed values on a fixture series:
- `trend.py` — SMA slope (L404) `slope = (sma - sma[d])/(d*sma)*1000`, `linreg` diff (L405), normalized; `d=max(round(S/4),2)`.
- `efficiency.py` — Kaufman ER (L468–480): `er_path=Σ|close[k]-close[k+1]|` over `er_period`; `er_net = (close-close[P])` (directional) or `abs(...)`; `er_val=er_net/er_path`.
- `vola.py` — ATR / StdDev / Intraday `(high-low)/close` SMA (L456–462) + `vola_pos = pir_of(vola, range_len)`.
- `momentum.py` — `price_ret=(close-close[L])/close[L]`; `mom_diverge=price_ret*vol_ret`; `mom_velocity=price_ret-price_ret[1]` (L443–453).
- `garch_har.py` — GJR (L159–177) and Garman-Klass+HAR (L179–188), with the exact constants (gjr α=0.03 β=0.90 γ=0.08; HAR 0.36/0.28/0.28; GK `0.5*lhl² − (2ln2−1)*lco²`).
- `pivots.py` — `ta.pivothigh/low(lb,lb)` (L491–492) + `calc_pivot_drift` (L142–151).
- [ ] One test file per feature; bounded-range + reference-value asserts. Commit each.

### Task 2.4: feature bank assembly + registry
- `bank.py`: `@register_feature` / `FeatureBankFactory`; `build_feature_matrix(df, cfg)->(X:DataFrame, feature_names)` over the configured dense grid (scales/periods/methods). Warm-up rows (insufficient history) → NaN, dropped/masked.
- [ ] Test: output shape, no inf, NaN only in warm-up, deterministic. Commit.

---

## Phase 3 — Label oracle 🟢

### Task 3.1: weighted multi-scale structural nest
- Create `src/cfd10/label_module/oracle.py`, tests.
- Interface: `OracleConfig(scale_nest:tuple[int,...], weight_curve:str, drawdown_pct:float, horizon:int, tau_strong:float, tau_regular:float)`; `label_turns(df, cfg)->DataFrame[top_score, bottom_score, top_tier, bottom_tier, top_weight, bottom_weight]`.
- Logic: candidate extremum (local max/min over nest's largest scale) → for each scale n in nest, confirm structural turn (price reverses ≥ drawdown_pct within horizon AND extremum is n-bar extreme) → weighted sum with `weight(n)` increasing in n (n≈2 ⇒ ~0); normalize → score; threshold → tier; score → sample weight.
- [ ] Test: synthetic V-bottom that is extreme at n=200 AND n=50 scores ≫ one extreme only at n=2. Test: monotone weight curve (`w(200)>w(50)>w(2)≈0`). Test: causal-truth flag — no label uses data beyond `horizon`. Commit.

### Task 3.2: label QA script
- `pipeline/qa_labels.py`: overlay oracle tops/bottoms on SPX 1D; assert the famous lows (2009-03-09, 2020-03-23, 2002-10, 2008-11) land in `strong` bottoms. Output a plot to `outputs/`. (Not a unit test — a sanity gate run during execution.) Commit.

---

## Phase 4 — CV harness 🟢 (LEAKAGE TEST IS THE POINT)

### Task 4.1: purged + embargoed walk-forward
- Create `src/cfd10/cv_module/splits.py`, tests.
- Interface: `purged_walk_forward(timestamps, label_horizon, embargo, n_folds, pooled_groups=None)->list[(train_idx,test_idx)]`.
- [ ] **Leakage unit test (critical):** for every fold, assert no `train_idx` falls within `[test_start - label_horizon, test_end + embargo]` of any test sample's label window. Construct a fixture where a naive split WOULD leak and assert this splitter does not. Test: per-asset grouping keeps an asset's bars from straddling train/test within the embargo. Commit.

---

## Phase 5 — Evaluation 🟢

### Task 5.1: event-based metrics (Hungarian + tolerance)
- Create `src/cfd10/eval_module/{events.py,metrics.py}`, tests.
- Interface: `match_events(pred_idx, true_idx, tolerance)->(tp,fp,fn,matches)` via `scipy.optimize.linear_sum_assignment` on |Δbars|≤tolerance; `event_prf(...)->(precision,recall,f1)`.
- [ ] Test: preds shifted by ≤tolerance count as TP; beyond tolerance count FP+FN; no double-matching. Known-case F1. Commit.

### Task 5.2: deflation + report
- `deflation.py`: deflated metric / PBO via CSCV over fold combinations. `report.py`: `Scorecard` dataclass → markdown table (per side/asset/era). Test: deflation reduces score under many trials; PBO∈[0,1]. Commit.

---

## Phase 6 — GBDT baseline 🟢 (the yardstick + first real OOS number)

### Task 6.1: LightGBM baseline through the CV harness
- Create `src/cfd10/teacher_module/baseline.py` (or `eval`), `pipeline/fit_baseline.py`, tests.
- Interface: `fit_gbdt_cv(X, y, weights, splits, cfg)->FoldResults`; binary per side; sample-weighted by oracle weight; probability output → threshold tuned on train fold only.
- [ ] Test: runs on a 1k-row fixture, returns finite OOS F1, respects splits (no leak — reuse Phase 4). 
- [ ] Execution gate: produce first **OOS scorecard** on SPX 1D. **Run-3 comparison baseline** comes from either (a) a one-time port of the rule detector's signals, or (b) a Pine export (Phase 8 / USER). Note in report which. Commit.

---

## Phase 7 — Teacher 🟡 (CPU smoke local; GPU train on Colab)

### Task 7.1: windowed dataset + TCN model
- Create `src/cfd10/teacher_module/{datasets.py,models/tcn.py,models/__init__.py}`, tests.
- `WindowDataset(X, y, w, window:int)->(B, window, F)`. `TCN(cfg)` two heads (top/bottom), small (≤~100k params), dropout+weightdecay. Registry `register_model`.
- [ ] Test (CPU): **overfit-a-batch** — on 32 samples, train loss → ~0 in <200 steps (proves wiring). Test: output shape, prob∈[0,1]. Commit.

### Task 7.2: train loop + calibration + beat-baseline gate
- `train.py` (AMP-ready, early-stop on OOS fold, seed-set), `calibrate.py` (isotonic/Platt). `pipeline/fit_teacher.py`.
- [ ] CPU smoke: 1 epoch on tiny data completes, checkpoint saved. **Real training: Colab.** Gate: teacher OOS F1 must exceed GBDT baseline to proceed to distillation; else ship GBDT. Commit.

---

## Phase 8 — Distillation + Pine export + PARITY 🟢 build / 🔴 verify

### Task 8.1: distill teacher → shallow student
- Create `src/cfd10/student_module/distill.py`, tests.
- `distill(teacher_probs, X, max_depth, max_rules)->StudentModel` (sklearn `DecisionTreeClassifier`/rule list mimicking teacher soft calls); report fidelity (student↔teacher agreement) + OOS.
- [ ] Test: student fits teacher on fixture to ≥ target fidelity; depth ≤ cap; uses ≤ K features. Commit.

### Task 8.2: export student → Pine
- Create `src/cfd10/student_module/export_pine.py`, tests.
- Walk the tree → nested `if/and` Pine v6; emit only the feature computations the tree uses (reusing the `ta.cum` PIR idiom). Output `.pine` + a JSON of (feature, threshold, path).
- [ ] Test: a 3-leaf tree exports to Pine source containing the exact thresholds; the emitted Pine parses (lint via a syntax check / structural assertion). Commit.

### Task 8.3: parity verification 🔴 USER-GATED
- Create `src/cfd10/parity_module/verify.py`, `parity_module/pine_export_contract.md`.
- `verify(python_signals_csv, pine_export_csv, tolerance=0)->ParityReport` — bar-by-bar diff of features and final signals; report first divergence + max abs feature delta (mirrors the old 0.14-mean/0.9-max divergence hunt).
- [ ] Test: identical CSVs → 0 divergences; injected 1-bar diff → caught.
- **🔴 NEEDS USER:** `pine_export_contract.md` specifies exactly what to export from TradingView (see "Pine export contract" below). Ping user here.

---

## Phase 9 — Per-TF runs + cross-TF pooling ablation 🟡

### Task 9.1: per-timeframe pipeline + ablation
- `pipeline/run_all.py` loops target timeframes; trains/distills/exports each. Cross-TF pooling toggled by `data.cross_tf_pool` with a one-hot `source_tf` feature; judged by OOS **on the daily target**. Produce comparative scorecard. Commit.

---

## Pine export contract (what I'll ask you for at Phase 8)

To prove Python↔Pine parity I'll need, **from TradingView for one (instrument, timeframe, preset) — start with SPX 1D, preset "SPX 1D 2026-05-24 V15 Run 3"**:
1. A CSV export of the chart's **OHLCV** for the exact bars (so we score identical data), and
2. The **Data Window `dbg_*` series** the indicator already exports (L628–658: `agreement_high_side`, `ph_confirms`, `er_val_high`, `pivot_drift_high`, `dbg_gate_pass_high`, `dbg_pivot_high_raw`, the per-vote `dbg_high_vote_*`, `baseline_pivot_high`, …) exported bar-by-bar, plus the LOW-side equivalents if you can add the symmetric exports.

I'll send precise click-by-click export steps when we reach Phase 8. (The existing `data/enriched/..._TV_V11.csv` is the V11 vintage — a useful structural anchor but not v15-exact.)

---

## Self-Review (against spec)

- **Spec coverage:** §4.1→P1, §4.2→P2, §4.3→P3, §4.4→P7, §4.5→P8.1, §4.6→P8.2-8.3, §5→P9, §6→P4+P5, §7→P5/P6 metrics, §8→P0.3/notebook, §9 layout→P0, §11 deliverables→P6/P8/P9. ✅ All spec sections have ≥1 task.
- **Placeholder scan:** parity-critical code inlined (P2.1); all other tasks specify exact files + concrete test contracts (TDD: the test is the spec). Remaining intentional gate: Run-3 comparison source (P6) + Pine exports (P8.3) — both explicitly USER/port-flagged, not silent. ✅
- **Type consistency:** `normalize`, `load_csv/load_pair`, `csum_close/sma_at/pir_for_scale`, `agreement`, `build_feature_matrix`, `label_turns`, `purged_walk_forward`, `match_events/event_prf`, `fit_gbdt_cv`, `distill`, `export_pine`, `verify` — names used consistently across phases. ✅
- **Order/deps:** P0→P1→P2→{P3,P4,P5 independent}→P6→P7→P8→P9. Feature parity (P2) precedes everything that consumes features. ✅

## Execution model (dynamic workflows)

Per user directive, phases execute via **dynamic Workflows** with TDD: each task = a subagent that writes the failing test, implements, runs `pytest` locally, and commits. Independent modules (P3/P4/P5) can fan out in parallel (worktree isolation or directory-partitioned); dependent phases run in sequence. The teacher's real training (P7) and parity verify (P8.3) are the only steps that leave this machine (Colab GPU / user Pine exports).
