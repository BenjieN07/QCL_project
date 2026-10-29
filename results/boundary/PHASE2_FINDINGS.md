# Capacity results and predeclared refinement

Two fresh five-layer versions of our original two-qubit circuit completed 1,500 updates each. Both pass the original-validation non-degradation guard. Neither loaded the old model's weights or optimizer moments.

| Fixed input scale | Selected balanced validation score | Selected endpoint maximum error |
|---|---:|---:|
| 1.0 | 0.0067273 | 0.05849 |
| 0.9 | 0.0022477013 | 0.02721426 |

The smooth five-layer trial's original-validation joint loss is 0.0038949363, versus Day 4's 0.0494098296. Collar-validation joint loss is 0.0006004662. It meets the endpoint goal, improves original validation, and was still improving at the 1,500-update budget. No new holdout has been read. This supports the tested bundle of capacity, coverage, scheduling, and smoother encoding, not a single-cause proof or broad seed guarantee.

Before independent evaluation, declare two same-model refinements from this exact saved update-1,500 checkpoint and Adam state. Both add 1,500 updates to update 3,000 with unchanged five-layer architecture, input scale 0.9, balanced training data, and targets. One keeps rate 0.003; the other uses 0.003/0.001/0.0003 in three 500-update blocks. These are actual continuations, not new initializations. Tighten their original-validation guard to the parent checkpoint's 0.0038949363. Compare by the same balanced validation score; initialization remains eligible. Freeze outcomes and selection before the first independent holdout read. Preserve earlier failures and controls unchanged.
