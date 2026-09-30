# Cache and event schemas

## Feature cache

`extract_features.py` writes an NPZ and companion JSON metadata. Neither is
included in the code archive. Arrays are `features` (N by D, float32),
`text_features` (C by D, float32), `labels` and `domains` (length N, integer),
`paths` (length N, string), `classes` (length C, string), and `domain_names`.
Metadata records dataset/backbone/backend, prompt, class/domain ordering, sample
count, weight SHA-256, cache SHA-256, extraction PyTorch version, and device.
Array row order is part of the source identity.

## Main event CSV (gzip)

One row represents one scored stream sample. `run_streams.py` writes
`STREAM__seedS__budgetB__STRATEGY.events.csv.gz` unless summary-only is requested.
The matching summary contains the SHA-256 of the compressed event file. Position,
original index, labels, and class predictions use zero-based integer indices.

| Field | Meaning |
| --- | --- |
| `position`, `original_index` | Stream position and source-cache row index. |
| `path`, `domain` | Source-relative sample identifier and domain index. |
| `label` | Evaluator's ground-truth class; not an adapter input. |
| `prediction`, `correct` | Frozen pre-intervention controller prediction and its 0/1 correctness. |
| `zero_shot_prediction`, `zero_shot_correct` | Raw-cache frozen dot-product prediction and its 0/1 correctness. |
| `confidence` | Probability of the controller's predicted class. |
| `entropy` | Mixture natural entropy normalized by ln(C). |
| `margin` | One minus the gap between the largest two mixture probabilities. |
| `disagreement` | Mean KL of expert distributions to the learned mixture, divided by ln(number of experts). Not generally JS divergence or bounded by 1. |
| `coverage` | Reciprocal square root of labeled count for the predicted class plus one. |
| `query_value` | Cube root of nonnegative entropy × disagreement × coverage. |
| `selection_score` | Score used by the selected query policy. |
| `selection_mode` | Chosen policy branch on queried samples, such as `query_value`, `risk_margin`, or `persistence_periodic`; otherwise empty. |
| `action` | Final logged action: `ask`, `adapt`, `predict`, or retained `pending` for non-updated samples in the zero-budget path. |
| `adapted`, `queried` | 0/1 indicators for pseudo adaptation and oracle query. |
| `oracle_label` | Revealed class for a queried sample, possibly corrupted by the specified noise simulation; empty otherwise. |
| `query_revealed_at` | Stream position when the queried label becomes available; empty otherwise. |

Rows contain evaluator labels for auditability. Their presence in the output
does not mean selection may use them. The controller predicts first. For
segment-based policies, query selection, label reveal, and unqueried pseudo
updates occur at segment close and affect future predictions only. A queried
sample is not also pseudo-adapted. The dynamic-entropy comparator can query
earlier within a segment under its separate one-query-per-segment rule.
Class-correlated streams use labels to construct a synthetic order before replay;
that construction is not an online label observation by the adapter.

## Summary JSON and indexes

Summary fields include run identity, stream/seed/strategy, sample and exact
budget/query counts, configuration, oracle-noise settings/counts, online and
zero-shot accuracy, their difference, Worst-500, harmful adaptation rate, ECE,
pseudo-memory statistics, final state, and event hash. Accuracy values are
fractions, not percentages. Worst-500 is the minimum mean over all consecutive
500-sample windows (or the full sequence if shorter). Harmful adaptation rate is
the fraction of all samples for which raw zero-shot is correct and the adapted
prediction is wrong. ECE uses 15 equal-width confidence bins.

`event_sha256` is null for summary-only runs. `run_review_supplement.py` and
`run_threshold_review.py` use summary-only mode. Main CLI runs also write a
JSON-lines run index. Merging anchors must include the parent cache directory:
run identifiers alone are not unique across datasets/backbones.

## TDA traces and checkpoints

The CPU review runner stores one JSON provenance/summary and NPZ trace per
cache/stream/seed/mode. Traces contain `original_index`, `prediction`,
`zero_shot_prediction`, `correct`, `zero_shot_correct`, and `confidence`.
Provenance includes feature-cache and algorithm hashes, NumPy version, entropy
mode, fixed parameters, and the audited official source commit. Resuming rejects
stale provenance and missing/corrupt traces.

The CUDA runner writes `CACHE__STREAM__seedS.npz` with `original_index`,
`prediction`, and `confidence`, plus per-run summaries and a device/runtime
manifest. Its targets are used for evaluation only after predictions finish.
The GPU aggregation mode recomputes correctness from the cache, verifies stream
orders and source hashes, and records CPU/GPU prediction disagreements.
