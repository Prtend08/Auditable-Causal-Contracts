# Auditable Causal Contracts for Streaming Recognition

Reference implementation for *Auditable Causal Contracts for Budgeted Selective
Supervision in Streaming Visual Recognition*. Ask-or-Adapt is the controller
and project name. This public repository contains the paper's source,
protocols, requirements, an event schema, and one
small deterministic audit fixture. Benchmark images, weights, feature caches,
prediction arrays, and the full benchmark event archive are not redistributed.

## Start here

### Run the audit fixture first

The fixture is a complete 120-sample synthetic stream with one delayed query.
It demonstrates the exact event-file hash and the per-run summary fields used
by the audit code:

```bash
python examples/audit_fixture/verify_audit_fixture.py
```

The command checks the event count, sequential positions, query count, online
accuracy, and the SHA-256 recorded in `summary.json`. The fixture is not a
benchmark result; it is a reproducible audit demonstration.

Use tag `v1.0.1` for the manuscript release. It preserves file bytes across
Git checkouts so the fixture SHA-256 is unchanged on Windows and Linux. The
full benchmark event archive and its per-run hash manifest are not included;
the fixture verifies the audit procedure, not all reported benchmark results.
`RELEASE_VALIDATION.json` records the earlier code-only package validation.
The root `_Code_20260930.zip` is retained as a legacy archive; use the
expanded source tree and fixture for this release.

- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): CPU/GPU commands and numerical limitations.
- [EVENT_SCHEMA.md](EVENT_SCHEMA.md): cache, event, summary, and TDA trace schemas.
- `src/a2a/core.py`: four-expert controller and memory updates.
- `src/run_streams.py`: streaming simulation, exact budgets, delayed labels, baselines, and ablations.
- `src/run_tda_review_matrix.py`: corrected, resumable NumPy TDA replay and GPU/CPU comparison.
- `src/run_tda_cuda_batch.py`: FP64 CUDA TDA replay, batching independent streams.
- `src/run_threshold_review.py`: post-hoc one-at-a-time threshold diagnostics.
- `src/analyze_revision_comparisons.py`: archived-event comparisons and canonical raw zero-shot audit.
- `src/analyze_paired_anchor_ci.py`: post-hoc paired anchor intervals and the audit of unavailable five-percent score-repair evidence; requires separately supplied result artifacts.
- `src/run_budget_leakage_review.py`: new same-runtime, full-event 1%/5% budget replay (180 runs); uses frozen controller defaults and separately supplied six feature caches.
- `src/audit_budget_leakage_replay.py`: independent read-only recheck of the 180 event traces and optional transfer archive; writes only its audit report and requires separately supplied inputs.
- `tests/`: synthetic unit tests; no benchmark download needed.
- `config/review_round2.yaml`: review-run settings and interpretation limits.
- `protocol/`: preserved historical local protocol files.
- `scripts/`: historical Linux launchers, retaining deployment paths that must be adapted.
- `environment/`: replay requirements and archived environment records.
- `CODE_MANIFEST.sha256`: hashes of packaged files, excluding this manifest itself.

For the paper's benchmark claims, run the scripts under `scripts/` after
supplying the datasets, model weights, and frozen feature caches described in
`REPRODUCIBILITY.md`. The expected output is a per-run summary CSV plus event
files whose SHA-256 values can be checked with the same audit pattern as the
fixture.

The `protocol/preregistration_v1.yaml` filename and fields such as
`registered_at_local` are historical local documentation. They do not establish
public preregistration, independent timestamping, or an external registry entry.
Do not reinterpret the review additions as prospectively confirmed tests.

## Data and model sources

The links below identify upstream sources, not bundled downloads. Dataset and
model licenses/terms remain those of their respective distributors. Download
availability, exact image inventory, and checkpoint identity must be checked
when reproducing an experiment.

- [DomainBed](https://github.com/facebookresearch/DomainBed) supplies preparation
  code for PACS, OfficeHome, and the TerraIncognita subset. The upstream repository
  is archived/read-only as of this release.
- [OfficeHome parquet snapshot](https://huggingface.co/datasets/flwrlabs/office-home)
  is supported by `OfficeHomeParquet`. Its recorded 15,588-row inventory differs
  from some other distributions of OfficeHome. Do not silently interchange them.
- [OfficeHome original project](https://www.hemanthdv.org/officeHomeDataset.html)
  and [original code](https://github.com/hemanthdv/da-hash).
- [OpenAI CLIP](https://github.com/openai/CLIP): ViT-B/16 backend.
- [OpenCLIP](https://github.com/mlfoundations/open_clip): ViT-L/14 backend.
- [Official TDA](https://github.com/kdiAAA/TDA), audited at commit
  `e697fb0c8078cdeff93daa56bcf8860702542069`.

DomainBed's documented preparation entry point is:

```bash
git clone https://github.com/facebookresearch/DomainBed.git
cd DomainBed
python -m domainbed.scripts.download --data_dir ./data
```

This command can download additional DomainBed benchmarks. Inspect upstream
download options and storage requirements before running it. Our folder reader
expects `DATA_ROOT/domain_name/class_name/image.jpg`; names must match the
explicit lists in `src/extract_features.py`. The extraction program records
checkpoint and cache SHA-256 hashes. No pretrained checkpoint identifier is
substituted for the user's actual weight file.

## Scope and cautions

The study replays fixed CLIP/OpenCLIP features with a simulated oracle using
benchmark annotations. It is not a human-participant study or an end-to-end
image-model adaptation benchmark. TDA and raw zero-shot use no oracle labels;
Ask-or-Adapt and Ask-only use a 1% label budget in the review comparison. Their
comparison is not equal-budget. Ask-only is a supervised-only adaptation anchor,
not an oracle returning correct current-sample predictions.

TDA is a single-view cache-equation replay with fixed parameters. Its current
entropy gate follows the official code (`H_ln / log2(C)`), while retained legacy
TDA scripts use `H_ln / ln(C)`. See the reproduction guide before selecting a
runner. The raw zero-shot comparator is reconstructed from original raw-cache
predictions, not the controller's renormalized no-update mixture.

The current controller deliberately retains the archived arithmetic. Numerically
near-zero KL disagreement can change query selection across NumPy/BLAS runtimes,
then change later adaptation trajectories. Exact historical trajectory replay is
not guaranteed merely by using the same seeds. Same-runtime paired diagnostics
and explicitly checked event traces are the supported evidence.

Historical analysis scripts can generate figures when run; generated figures
are not part of this archive. Paths and omitted inputs referenced by historical
scripts are not evidence that those inputs are bundled. This code-only release
does not by itself reproduce archived paper tables without the matching data,
weights, cached features, and/or separately supplied event artifacts.

No new project-wide software license is assigned by packaging. The authors
should choose and supply an appropriate project license before public
redistribution. Upstream components retain their own licenses.
