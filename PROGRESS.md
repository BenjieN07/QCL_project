# Progress and review gates

| Phase | Status | Evidence |
|---|---|---|
| Day 1: setup, data, circuit | IN PROGRESS | Configuration and dependency manifest |
| Day 2: training loop | NOT STARTED | — |
| Day 3: fit and baselines | NOT STARTED | — |
| Day 4: depth and seed study | NOT STARTED | — |
| Day 5: shot study | NOT STARTED | — |
| Day 6: optional hardware | NOT STARTED | Requires explicit consent |
| Day 7: report | NOT STARTED | — |

## 2026-10-28 — checkpoint 1

Created the project specification, pinned PennyLane version, configuration, and review workflow. The repository began empty. System `python` is an unusable Windows alias; a project virtual environment is being created using the bundled Python runtime. No numerical results yet. QPU usage: zero.

Next: deterministic disjoint datasets and target figure; then circuit validity benchmark. Stop after Day 1.

## 2026-10-28 — checkpoint 2

Dataset checks passed: seed 2026 reproduces 48 training, 16 validation, and 24 test inputs, all distinct across splits. Target identity sin² + cos² passed. Saved `results/day1/data.npz`, `data_checks.json`, and `targets.png`. Installed dependencies in the project environment and saved exact versions to `requirements-lock.txt`; `pip check` found no broken requirements. Early attempts to run preparation before installation finished failed due to missing Matplotlib; the completed run passed.

GitHub push failed with invalid username/token. Commits remain local until authentication is repaired. No hardware access or training performed.
