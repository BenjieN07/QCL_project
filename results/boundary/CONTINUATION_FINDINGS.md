# Phase 1 findings before capacity follow-up

The four declared 3-layer continuation branches each completed 750 additional Adam updates from the same Depth study checkpoint and optimizer state. No new holdout was read. All failed the endpoint-error goal of 0.05.

| Approach | Final original-validation loss | Final collar-validation loss | Final endpoint maximum error | Final iterate passes original-validation guard? |
|---|---:|---:|---:|---|
| Constant rate | 0.04673481 | 0.11279368 | 0.30287156 | Yes |
| Rate decay | 0.04785396 | 0.13007325 | 0.32545583 | Yes |
| Boundary-balanced | 0.05844260 | 0.01557404 | 0.14715017 | No |
| Boundary-balanced + smoother encoder | 0.06986137 | 0.00689505 | 0.11291340 | No |

The guard is original-validation joint loss <=0.04940982963545365. Earlier eligible checkpoints are retained where present. The smooth branch had no eligible checkpoint. These results suggest that more updates or this scheduling change alone are insufficient at this configuration. Explicit boundary training helps boundary validation but trades off original validation. They do not prove a fundamental limitation or isolate a single cause.

Following this measured result, declare a separate capacity experiment: two fresh five-layer versions of our existing two-qubit Rot/CNOT circuit, same seed 41, same 48+12 balanced training data and ten collar-validation points, differing only in fixed input scale (1.0 versus 0.9). Train 1,500 updates each at rates 0.03/0.01/0.003 for 500 updates each. No old weights or Adam moments are loaded into these larger architectures. Same validation score and guard; no changes to the existing four trials or their results.

The independent stratified holdout remains reserved and unread until this follow-up and validation-only selection are frozen. This is further validation-driven exploration, not a claimed confirmation experiment or a broad seed study.
