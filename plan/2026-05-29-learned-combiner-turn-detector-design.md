# Design Spec — Learned-Combiner Turn Detector (cfd10)

- **Date:** 2026-05-29
- **Status:** Draft for review
- **Author:** (kubens21@gmail.com) + Claude
- **Related:** `plan/cfd10-system-understanding.md` (system understanding of the existing Speculatores indicator + data corpus)
- **Predecessor:** `github.com/Sovenski/cfd9` (the old pipeline — intentionally retired; only the *labeling concept* is salvaged)

---

## 1. Problem & Goal

The existing system (`pine/speculatores_v15_presets_gold.pine`, 663 lines) is a rule-based market top/bottom detector whose decision layer is **hand-tuned**: per-side integer vote counts, hard gates, cooldown, edge-voting, and a ~60-parameter nested-ternary preset table found by an external black-box optimizer. The current best preset is **`SPX 1D 2026-05-24 V15 Run 3`**.

**Goal:** Replace the hand-coded decision layer with a *learned* one, while **keeping the entire feature library** as the input representation, and **end with a clear ruleset that ports back into Pine and reproduces the same signals on the same data.**

Concretely:
- Keep the Speculatores **features** (multi-scale PIR, trend, duration, volume surge, momentum/velocity, volatility by ATR/StdDev/Intraday, efficiency ratio, GJR asymmetry, HAR volatility, pivot drift, scale divergence).
- Drop the **decision layer** (vote-counting, hard gates, cooldown, edge-voting, preset ternaries).
- Learn how features interact via a **deep temporal teacher**, then **distill** it to a **shallow rule student** that is Pine-portable.

## 2. Background (one paragraph)

Speculatores runs two independent mirror pipelines — HIGH (tops) and LOW (bottoms). Each stacks a multi-scale "position-in-range" (PIR) agreement vote, a stack of hard gates, a count of soft confirmation votes, and a cooldown (`pivot_high/low`, Pine lines 585–586). Its features are mostly **dimensionless and bounded** (PIR ∈ [0,1], efficiency ratio ∈ [0,1], normalized slopes, volatility-position ∈ [0,1]), which is what makes them comparable across instruments and timeframes. The numerically-delicate parts (the `ta.cum(close)` "parity shim", Pine lines 73–113) exist to keep Pine bit-identical to a Python twin. Full detail: `plan/cfd10-system-understanding.md`.

## 3. Approach — teacher → distilled student

**Chosen:** *Deep teacher → distilled rule student.* DL discovers the best achievable signal; a compact ruleset deploys it.

### 3.1 The two-gap parity principle (critical — must stay true throughout)

There are two transformation gaps, and only one is lossy:

| Gap | Lossy? | Meaning |
|-----|--------|---------|
| Teacher (deep net) → Student (rules) | **Yes (~90-something % signal agreement)** | The shallow ruleset *approximates* the net. This is the accepted price of readability + portability. |
| Student (rules) → Pine | **No (exact)** | The student is a deterministic tree/rule-list over dimensionless features. Re-implemented in Pine with identical feature math, it reproduces signals **bit-for-bit on the same data.** |

So **"same signals in Pine on the same data" is guaranteed for the deployed student.** "Same signals as the deep net" is approximate by construction. The implementation MUST preserve this: the student may only use features and operations that Pine can reproduce exactly.

### 3.2 Why this is the right ML shape for the data

The dataset is small (SPX 1D ≈ 25k bars, only ~2k–14k "modern regime"; true structural turns number dozens–hundreds). Strong hand-crafted features + a learned combiner is the small-data-friendly form of deep learning: the hard representation work is already done, so the model only learns the interaction. A **GBDT baseline (LightGBM)** runs alongside the net as the honest yardstick — if the net cannot beat GBDT out-of-sample, we ship the simpler model.

## 4. System architecture

```
 GDrive data (raw_v16, == local) ──select(instruments, timeframes)──┐
                                                                     ▼
 [1] DATA LAYER  normalize schema (volume/Volume), neg-epoch handling, drop SI1! dup, align/resample
                                                                     ▼
 [2] FEATURE BANK  parity-faithful port of Speculatores features, dense across scales/periods/methods
        │                                                            │
        ▼ (per-bar features)                                         ▼ (windowed: W bars × features)
 [3] ORACLE  weighted multi-scale structural nest                [4] TEACHER (deep, GPU): TCN/CNN→LSTM→(Transformer)
     → graded score → Strong/Regular tiers + sample weights            two heads TOP/BOTTOM, calibrated probs
        │                                                            + GBDT baseline (per-bar, the yardstick)
        └──────────────► labels (top/bottom) ──────────────────────────┘
                                                                     ▼
 [5] DISTILL → STUDENT  shallow decision tree / rule list mimicking teacher probabilities
                                                                     ▼
 [6] EXPORT .pine  +  PARITY VERIFY (Python vs Pine, bar-by-bar)  →  per-timeframe rulesets
```

### 4.1 Data layer (`src/data_module/`)
- Load CSVs from the configured data root (GDrive path on Colab, local path otherwise).
- **Schema normalization:** `Volume`→`volume`; enforce `time,open,high,low,close,volume`.
- Parse **negative pre-1970 unix epochs** (1871 S&P composite) correctly; do not feed unsigned/ms parsers.
- **Drop the confirmed duplicate** `COMEX_DL_SI1!, 1_8d38f.csv` (it is a 2-bar subset of `…6bcd3.csv`).
- Decode raw_v16 names `{EXCHANGE}_{TICKER}, {minutes}_{hash}.csv` (1, 60, 240, 1D).
- Optional resample / cross-timeframe assembly for the pooling ablation (§5).
- **Anti-leakage is enforced downstream in CV (§6), not here**, but the loader tags every row with `(instrument, timeframe, timestamp)` so purging/embargo can operate.

### 4.2 Feature bank (`src/feature_module/`)
- Vectorized (NumPy/Numba) port of each Speculatores feature, matching Pine semantics **including the `ta.cum` stateless-SMA trick** so Python↔Pine agree.
- **Dense bank** (≈50–150 features): PIR over a scale grid; multi-scale agreement counts/ratios; trend slope + linreg-slope at several detect-scales; duration-at-extreme proxies; volume-surge ratios; momentum & momentum-velocity (Trend and Reversal); volatility via {ATR, StdDev, Intraday} × several lengths + volatility-position; efficiency ratio (directional & absolute) at several periods; GJR-asymmetry; HAR; pivot-drift; scale-divergence.
- All features **dimensionless / bounded** wherever possible (so no learned scaler must be ported to Pine; any necessary standardization constants are folded into exported student thresholds).
- **Sanity anchor:** cross-check overlapping features against `data/enriched/SPX_1D_…_TV_V11.csv` columns (conceptual, since enriched is the V11 vintage).
- Registry pattern: `register_feature(name)` → `FeatureBankFactory`.

### 4.3 Label oracle (`src/label_module/oracle.py`)
- **Salvaged + extended** from the old Scorer-v4 "structural nest": for each candidate extremum, test confirmation across a **nest of lookback scales** (e.g. {20, 50, 100, 200, …}); **weight increasing with scale** so structural (large-n) hits dominate and trivial (n≈2) hits contribute ≈0.
- Summed → a **graded structural-ness score**; thresholds τ_strong / τ_regular → `{none, regular, strong}` tiers (mirrors the enriched `Strong/Regular High/Low` split). The score also serves as the **per-sample training weight**.
- Computed independently for tops and bottoms.
- Forward-looking (it is ground truth) but used only as a training target; **CV purging/embargo prevents leakage** into the model's causal features.
- **Knobs (Hydra):** scale nest, weight curve, drawdown-% / horizon defining a "real" turn, τ thresholds. We tune these and report sensitivity.

### 4.4 Teacher + baseline (`src/teacher_module/`)
- **Input:** window of `W` bars × feature-bank dims (lets the model learn duration/edge/cooldown dynamics the rules hand-coded).
- **Default arch = 1D-CNN / TCN** (smallest capable model first); registry allows LSTM/GRU and a small Transformer if OOS justifies escalation.
- **Two heads** (top, bottom) or two models. **Loss:** focal / class-weighted, multiplied by oracle sample weights. **Calibrated** probabilities (needed for thresholds + distillation targets).
- Heavy regularization (dropout, weight decay, early stopping on the OOS fold); small capacity by design.
- **GBDT baseline (LightGBM)** on the per-bar features (no window) runs alongside as the yardstick + feature-importance readout.

### 4.5 Distillation → student (`src/student_module/distill.py`)
- Teacher emits **soft probabilities on all bars** (incl. unlabeled) → a dense supervised signal.
- Fit a **shallow decision tree / rule list** (depth/#rules capped for readability) to mimic the teacher (regression on logits or classification on teacher hard-calls).
- Report **fidelity** (student vs teacher agreement) and **OOS** (student vs oracle); trade depth ↔ fidelity explicitly.

### 4.6 Pine export + parity (`src/student_module/export_pine.py`, `src/parity_module/verify.py`)
- Emit student as Pine: the nested `if/and` rules + only the handful of feature computations they reference, as a new preset/branch or a standalone indicator.
- **Parity harness:** run student in Python and Pine on the same CSV, diff signals bar-by-bar (reuse the `dbg_*` data-window export discipline). **Target: exact match** (deterministic tree on Pine-expressible features).

## 5. Per-timeframe specialization & data selection

- Pipeline is **config-driven on which (instrument, timeframe) pairs to learn from.**
- Produce **one specialized teacher→student per target timeframe** (1D, 4h, 1h, 1m, …). In Pine you select the ruleset matching your chart timeframe (mirrors today's preset selector UX).
- **Default training plan for the daily model:** primary target **SPX 1D**, trained on the **pooled daily corpus** (~45 1D instruments) for more turn examples + regularization; model-selected and OOS-reported primarily on SPX 1D; other assets are a cross-asset generalization test.
- **Cross-timeframe pooling ablation (off by default):** optionally mix 1h/4h bars into the daily training set, justified by feature scale-invariance, with a one-hot/embedding **source-timeframe feature**, judged **purely by OOS on the daily target.** Measured, not assumed.

## 6. OOS / anti-overfitting protocol (non-negotiable, `src/cv_module/`)

Given the existing system's documented IS→OOS collapse, this is a first-class component:
- **Purged + embargoed walk-forward CV** (López de Prado): no training bar whose forward label-window overlaps a test bar may be used. This is the dominant leakage source in time-series ML.
- **Event-based metrics** with a tolerance window (Hungarian matching, like the old Scorer v3): precision / recall / F1 **on turns**, never per-bar accuracy.
- **Deflated** metrics + an overfitting-probability estimate (PBO / CSCV).
- **Pooled training; per-asset and per-era holdouts** to prove the model didn't just memorize 2009 / COVID.
- **HIGH and LOW reported separately**, with the explicit stance that **HIGH may remain a noise floor** — the system must report this honestly, not overfit a top signal into existence.

## 7. Metrics & success criteria

- **Primary:** event-based F1 (with tolerance) per side, **deflated**, evaluated OOS on SPX 1D, **vs the Run-3 baseline** reproduced in the same harness.
- **Secondary:** precision@k (few high-confidence calls), cross-asset OOS, per-era stability (à la the "local_mean/best" stability the old logs tracked).
- **Distillation fidelity:** student↔teacher signal agreement ≥ a target we set during design of the student (e.g. ≥90%); **student↔Pine = exact (required).**
- **Success = the distilled SPX-1D LOW ruleset beats Run-3 LOW on deflated OOS**, and HIGH is characterized honestly (improved or confirmed-unlearnable). We do not declare success on IS gains.

## 8. Infrastructure & execution environment

**Pattern (mirrors the retired `cfd9` workflow):** a user-uploaded Colab notebook mounts Google Drive (data) and clones the GitHub repo (code).

- **Compute:** Colab, **T4 or L4 GPU** (small temporal net on daily bars; A100/H100/TPU would burn credits for ~no speedup). 12 GB RAM is sufficient — the full CSV corpus is ~100 MB.
- **Data:** Google Drive folder holding **the same `raw_v16` data that exists locally** (exact copy). Notebook reads from a configured GDrive path; data is **never committed to git.**
- **Code:** a **new GitHub repo created on the user's account**, cloned into Colab.
  - **Proposed repo:** `github.com/Sovenski/cfd10` — **private** (trading research; should not be public). *← confirm name + visibility before creation.*
  - Repo creation is an **implementation-phase action** (not done during spec authoring).
- **Notebook skeleton** (the uploaded `.ipynb`):
  ```python
  # Cell 1 — Mount Google Drive
  from google.colab import drive
  drive.mount('/content/drive')

  # Cell 2 — Clone/update repo (master) and install deps
  import os
  from pathlib import Path
  REPO_DIR = Path('/content/cfd10')
  REPO_URL = 'https://github.com/Sovenski/cfd10.git'
  if not REPO_DIR.exists():
      !git clone {REPO_URL} {REPO_DIR}
  %cd {REPO_DIR}
  !git pull --ff-only -q
  !pip install -q -r requirements.txt
  !git rev-parse --short HEAD

  # Cell 3 — Configure run (Hydra overrides: data path on GDrive, target timeframe, train set, oracle knobs)
  # Cell 4 — build_features  → Cell 5 — fit_teacher (+GBDT)  → Cell 6 — distill
  # Cell 7 — export_pine + verify_parity  → Cell 8 — OOS scorecard + plots
  ```
- **Private-repo cloning in Colab:** uses an HTTPS token (GitHub PAT) supplied at runtime via Colab secrets / env var — **never hard-coded in the notebook or repo** (per security rules). The exact mechanism is an implementation detail to finalize at setup.

## 9. Repo layout & coding standards (matches user CLAUDE.md)

```
cfd10/
├── conf/                      # Hydra configs (dataclass-backed, frozen)
│   ├── config.yaml
│   ├── data/  oracle/  features/  teacher/  distill/  cv/
├── src/
│   ├── data_module/           # loaders, schema norm, resample
│   ├── feature_module/        # parity-faithful feature bank (registry)
│   ├── label_module/          # weighted multi-scale oracle
│   ├── cv_module/             # purged/embargoed walk-forward splits
│   ├── teacher_module/        # temporal nets + GBDT baseline (registry)
│   ├── student_module/        # distillation + Pine export
│   ├── eval_module/           # event metrics, deflation, reports
│   ├── parity_module/         # Python↔Pine verification harness
│   └── utils/                 # seeding, logging, env recording
├── pipeline/                  # Colab entry scripts (build/fit/distill/export/verify)
├── tests/                     # unit tests (feature parity, oracle, CV no-leak)
├── plan/                      # specs & plans (this file)
├── requirements.txt
├── pyproject.toml             # uv-managed
└── .gitignore                 # excludes data/, outputs/, .env, settings.json, *.pt, secrets
```
Standards: PEP 8, type hints required, `@dataclass(frozen=True)` configs, factory+registry per module, files 200–400 lines, `logging` not `print`, `set_seed()` + environment recording for reproducibility, `__all__` in every `__init__.py`.

## 10. Configuration (Hydra)

Single composed config; CLI/notebook overrides select everything: `data.train_pairs`, `data.target_pair`, `data.cross_tf_pool`, `oracle.*`, `features.*`, `teacher.arch`, `cv.*`, `distill.max_depth`. Hydra auto-saves resolved config + overrides to `outputs/` for reproducibility.

## 11. Deliverables

1. Config-driven pipeline runnable end-to-end on Colab: `build_features → fit_teacher → distill → export_pine → verify_parity`.
2. **Per-timeframe Pine rulesets** + a **parity report** proving Python↔Pine signal equality (exact for the student).
3. An **OOS scorecard** (per side / asset / era, deflated) vs Run 3.
4. The uploaded **Colab notebook** + the **GitHub repo** + a `requirements.txt`.

## 12. Risks, caveats & open questions

- **HIGH side may be unlearnable** in this feature space (Run-3 HIGH OOS = 0.000). Expectation managed; the model reports it honestly.
- **Tiny labels** → overfitting is the central enemy; the purged-CV + deflation + GBDT-yardstick design is the mitigation. If even GBDT can't beat Run-3 OOS, that is itself a (valuable, honest) result.
- **Distillation gap:** the deployed ruleset is weaker than the net by design; we report exactly how much.
- **Cross-TF pooling** may inject intraday-seasonality noise; gated behind an ablation.
- **Feature parity:** any feature that can't be reproduced exactly in Pine is barred from the student (even if the teacher may use it).
- **Open:** exact scale-nest / weight curve for the oracle (tuned empirically); final student family (single tree vs short rule list); whether one daily model + source-TF feature beats per-TF models.
- **Open (setup):** repo name + visibility confirmation; private-repo token mechanism in Colab.

## 13. Out of scope (YAGNI)

- Live trading / execution / brokerage integration.
- Real-time streaming inference (research/backtest cadence only).
- Porting the deep net itself into Pine (explicitly rejected — distillation is the bridge).
- Re-deriving or maintaining the old `cfd9` optimizer.

## 14. Milestones (high-level; detailed plan via writing-plans)

1. Repo + Colab harness + data loader + schema norm (runs end-to-end on a stub).
2. Feature bank + **parity test** vs Pine on SPX 1D.
3. Oracle + label QA (visual check vs famous SPX turns).
4. CV harness with a **leakage unit test**.
5. GBDT baseline + OOS scorecard vs Run 3.
6. Teacher (TCN) + calibration; beat-the-baseline gate.
7. Distillation → student + fidelity report.
8. Pine export + **exact parity verification**.
9. Per-timeframe runs + cross-TF pooling ablation.
