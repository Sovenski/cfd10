# cfd10 — Learned-Combiner Market Turn Detector

Replaces the hand-tuned voting layer of the Speculatores Pine indicator with a
**learned** combiner: keep the feature library, train a deep temporal *teacher*,
then **distill** it into a shallow rule *student* that ports back into Pine and
reproduces the same signals on the same data.

## Status
Early scaffold. See the design + plan:
- `plan/2026-05-29-learned-combiner-turn-detector-design.md` — design spec
- `plan/2026-05-29-cfd10-implementation-plan.md` — phased implementation plan
- `plan/cfd10-system-understanding.md` — understanding of the original system

## Layout
- `src/cfd10/` — pipeline modules (data, features, oracle, cv, eval, teacher, student, parity)
- `pipeline/` — Colab entry scripts
- `notebooks/colab_cfd10.ipynb` — uploaded Colab notebook (mounts Drive, clones this repo)
- `pine/` — the original Speculatores indicator (parity reference)
- `tests/` — pytest suite

## Local dev
```bash
uv venv --python 3.12
uv pip install -r requirements-dev.txt
pytest
```
Data is **not** in the repo. Locally it sits in `data/` (gitignored); on Colab it is
mounted from Google Drive.

## Compute
Local CPU builds and tests everything except teacher training, which runs on a
Colab T4/L4 GPU via the notebook.
