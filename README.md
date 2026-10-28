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

## Scientific context

Inspired by [Mitarai et al., Quantum Circuit Learning (2018)](https://arxiv.org/abs/1803.00745), rather than an exact reproduction. Implementation follows the [PennyLane PyTorch interface](https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html).
