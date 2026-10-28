# Quantum Circuit Learning: sine and cosine

An incremental educational experiment. No quantum advantage claim. Local simulator only; IBM hardware requires separate explicit consent.

## Review checkpoints

We stop after each day's exit gate. Day 1 establishes data and circuit validity; it does not train a model. See [the original roadmap](QCL_Progress_Report_Codex_Plan.md) and [current progress](PROGRESS.md).

## Setup (PowerShell, Python 3.11+)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Exact installed versions are saved with benchmark results. Configuration and split seeds are tracked in `config.json`.

## Run and review Day 1

```powershell
.\.venv\Scripts\python.exe -m qcl.prepare
.\.venv\Scripts\python.exe -m qcl.benchmark
```

Open `results/day1/REVIEW.md` for the review checklist, figures, and encoding description. `data_checks.json` and `benchmark.json` contain machine-readable pass/fail evidence, seeds, versions, gate counts, parameter sensitivity, and a timing measurement. A failed assertion exits with an error. Timings describe only this machine and batch, not quantum speedup.

For a single review command, run `.\review-day1.ps1`. It rebuilds the figures and checks, then prints the review note. For exact environment recreation use `pip install -r requirements-lock.txt` after creating the virtual environment.

The fixed splits are 48 training, 16 validation, and 24 test inputs. Validation is reserved for architecture selection; test labels are not used by the Day 1 circuit benchmark. The plotting grid has 101 points. Day 2 will add training and loss history after this review gate.

## Scientific context

Inspired by [Mitarai et al., Quantum Circuit Learning (2018)](https://arxiv.org/abs/1803.00745), rather than an exact reproduction. Implementation follows the [PennyLane PyTorch interface](https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html).
