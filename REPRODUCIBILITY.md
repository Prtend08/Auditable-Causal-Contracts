# Reproduction guide

## Environments

Use separate environments for CPU replay and CUDA/feature extraction.

| Purpose | Recorded environment | Interpretation |
| --- | --- | --- |
| Historical feature extraction/main study | Archived dependency record lists NumPy 1.26.4, PyTorch 2.0.1+cu118, torchvision 0.15.2+cu118 | `environment/requirements-historical-autodl-lock.txt` is preserved provenance, not a newly verified clean-install lock or guarantee of the original BLAS binary. |
| Current review CPU replay | Python 3.12.14, NumPy 2.3.5, Windows 11 build 26200, scipy-openblas 0.3.30, one BLAS thread | Tested source/test runtime. CPU core and TDA replay require NumPy; other analysis/extraction tools need optional dependencies. |
| New paired-budget event replay | Python 3.10.8, NumPy 1.26.3, pandas 2.2.3, Linux, one BLAS thread | Recorded server runtime for the separate post-hoc 180-run experiment; not the historical summary-only trajectories. |
| Current review CUDA TDA | PyTorch 2.1.2+cu121, CUDA 12.1, NVIDIA GeForce RTX 4090 D, FP64, TF32 disabled | Batches independent stream conditions, not chronological samples. Python/NumPy versions should be recorded by the person reproducing this run. |

The review CPU environment also reports pandas 3.0.1, matplotlib 3.11.2,
seaborn 0.13.2, and pytest 9.1.1. Those observations are not a claim that every
optional historical script was tested under those packages. Package availability
and platform compatibility must be checked during installation. No clean-room
dependency installation is claimed for this archive.

```bash
python -m venv .venv
# Activate the environment using the command appropriate to your shell.
python -m pip install -r environment/requirements-replay.txt
python -m pytest -q
```

Install `environment/requirements-analysis.txt` only for the analysis scripts.
For CUDA, install the compatible PyTorch/torchvision wheels following the
[official PyTorch previous-versions instructions](https://pytorch.org/get-started/previous-versions/),
using the versions in `environment/requirements-cuda-reference.txt` as the
recorded reference. Install the selected CLIP backend from its official
repository separately. Do not install the historical full lock over an unrelated
current environment and assume bitwise equivalence.

Set thread limits before importing NumPy. Bash:

```bash
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
```

PowerShell equivalents:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:PYTHONPATH = "$PWD/src"
```

The following multiline commands use Bash continuations. In PowerShell put each
command on one line. Replace example paths with your own paths.

## 1. Build the six fixed caches

```bash
python src/extract_features.py --dataset PACS --data-root /data/PACS \
  --output /work/cache/pacs_openai_vitb16.npz --backend openai \
  --model-name ViT-B/16 --clip-source /code/CLIP --weights /weights/ViT-B-16.pt

python src/extract_features.py --dataset PACS --data-root /data/PACS \
  --output /work/cache/pacs_openclip_vitl14.npz --backend open_clip \
  --model-name ViT-L-14 --weights /weights/openclip-vitl14.pt
```

Repeat for `OfficeHome` and `TerraIncognita`. Use cache prefixes `officehome`
and `terra`. The six basenames must be:

```text
pacs_openai_vitb16.npz       pacs_openclip_vitl14.npz
officehome_openai_vitb16.npz officehome_openclip_vitl14.npz
terra_openai_vitb16.npz      terra_openclip_vitl14.npz
```

For the OfficeHome parquet distribution add `--data-format officehome_parquet`
and point `--data-root` at its shard directory. Keep the companion `.json`
metadata files. Check the exact sample inventories, class/domain ordering,
weight hashes, prompts, and cache hashes before comparing with archived runs.
The recorded inventories are PACS 9,991, OfficeHome 15,588, and TerraIncognita
24,788 samples per backbone. Re-extraction with different package versions or
different weights is a new feature-source condition, not an exact replay.

## 2. Main exact-budget matrix (CPU, 630 runs)

Run the following for each dataset/backbone cache, placing output under
`/work/results/confirmation/BACKBONE/DATASET` where BACKBONE is `openai_vitb16`
or `openclip_vitl14`, and DATASET is `PACS`, `OfficeHome`, or `TerraIncognita`:

```bash
python src/run_streams.py --cache /work/cache/pacs_openai_vitb16.npz \
  --output-dir /work/results/confirmation/openai_vitb16/PACS \
  --streams iid_random abrupt_domain class_correlated --seeds 10 11 12 13 14 \
  --budgets 1 --strategies random periodic entropy margin disagreement \
  dynamic_entropy_online ask_or_adapt_v2 --pseudo-confidence 0.80 \
  --pseudo-agreement 0.75 --persistence-switch-rate 0.35 --risk-entropy-threshold 0.50

python src/analyze_v2_confirmation.py --root /work/results/confirmation \
  --output /work/analysis/confirmation
```

This yields the `all_confirmation_runs.csv` reference required by later
comparison runners. Main events are written unless `--summary-only` is supplied.
The dynamic-entropy comparator is a selection-only rule inside this controller
and exact-budget protocol, not a reproduction of an entire external method.
Historical `scripts/run_v2_confirmation.sh` encodes all six blocks but uses
old AutoDL paths; edit its deployment variables before running it.

## 3. Anchors and canonical raw zero-shot

```bash
python src/run_review_supplement.py --cache-root /work/cache \
  --output-dir /work/anchors --workers 4 --strategies zero_shot ask_only

python src/validate_caches.py /work/cache/*.npz \
  --output /work/results/multibench_cache_audit.json

python src/analyze_revision_comparisons.py \
  --confirmation-root /work/results/confirmation --cache-root /work/cache \
  --anchor-root /work/anchors \
  --reference-csv /work/analysis/confirmation/all_confirmation_runs.csv \
  --output /work/analysis/comparisons
```

The cache-audit JSON must be in the **parent of the confirmation results root**,
with exactly the filename shown. The wildcard above is a Bash glob; pass the
six explicit cache paths in PowerShell. This newly generated manifest checks
internal source consistency only. It does not establish identity with historical
paper caches unless compared to independently retained historical hashes.
Its bulk-dot-product accuracy is not a substitute for the canonical rowwise
zero-shot audit below. Step 2 also supplies `paired_statistics.csv` beside the
reference CSV; the revised audit requires both files.

The supplement runner deliberately writes **summary-only** anchors. It does not
provide anchor event logs or an independent event-level timing audit of those
anchor runs. Do not fabricate event hashes when they are absent.

The retained `zero_shot` policy is a no-update controller anchor that renormalizes
features/prototypes. It can differ slightly from the raw-cache zero-shot
prediction. The canonical frozen reference is the original rowwise
`argmax(features[i] @ text_features.T)` from main events. The revised comparison
audit reconstructs/checks that reference and its rolling windows independently.

## 4. Corrected TDA matrix (90 runs per backend)

After step 2 has produced the reference CSV, CPU:

```bash
python src/run_tda_review_matrix.py --cache-root /work/cache \
  --output-dir /work/tda_cpu --workers 4 \
  --main-csv /work/analysis/confirmation/all_confirmation_runs.csv --anchors /work/anchors
```

GPU (requires CUDA; no main-results CSV needed to run adaptation):

```bash
python src/run_tda_cuda_batch.py --cache-root /work/cache \
  --output-dir /work/tda_cuda --seeds 10 11 12 13 14
```

Aggregate GPU output, independently evaluate its traces, and compare it with all
available completed CPU checkpoints:

```bash
python src/run_tda_review_matrix.py --cache-root /work/cache \
  --gpu-dir /work/tda_cuda --output-dir /work/tda_comparison \
  --cpu-audit-dirs /work/tda_cpu \
  --main-csv /work/analysis/confirmation/all_confirmation_runs.csv --anchors /work/anchors

python src/run_tda_review_matrix.py --cache-root /work/cache \
  --output-dir /work/tda_audit --self-test
```

The released official TDA code divides natural-log softmax entropy by `log2(C)`
before the negative-cache gate. Our corrected mode is `official_log2`, the
default in both new runners. Uniform predictions therefore produce `ln(2)`,
not 1, under this **official-code convention**. The earlier local
`run_tda_cached.py` and `run_tda_cached_torch.py` retain the legacy `H_ln/ln(C)`
gate for historical comparison and must not be mixed with corrected results.
The new CPU runner supports `--entropy-mode legacy_ln` into a separate output
directory. Old seed-10 rows are not silently reused for the corrected matrix.

Source: [official utility](https://github.com/kdiAAA/TDA/blob/e697fb0c8078cdeff93daa56bcf8860702542069/utils.py),
[runner](https://github.com/kdiAAA/TDA/blob/e697fb0c8078cdeff93daa56bcf8860702542069/tda_runner.py), and
[fixed settings](https://github.com/kdiAAA/TDA/blob/e697fb0c8078cdeff93daa56bcf8860702542069/configs/imagenet.yaml).
The replay uses single cached views and the study's fixed prompts/features. It
does not rerun original image augmentation, feature extraction, dataset-specific
tuning, or the official end-to-end benchmark. The current unlabeled feature may
enter the TDA cache before its adapted prediction, matching the official order.
No current or future ground-truth label enters that adapter.

## 5. Post-hoc threshold diagnostics

```bash
python src/run_threshold_review.py --cache-root /work/cache \
  --output-dir /work/thresholds --workers 4 \
  --reference-csv /work/analysis/confirmation/all_confirmation_runs.csv
```

This is seven one-at-a-time configurations, six caches, three streams, seed 10,
and a 1% budget: 126 runs, including 18 freshly replayed defaults. It does not
select a new threshold and is not a confirmatory sweep. The runner may exit
nonzero **after saving all outputs** when defaults do not match the supplied
archival reference. Inspect its verification fields instead of treating that
exit as proof that no runs completed. Compare variants to their same-runtime
replayed defaults.

The four initial expert distributions can coincide mathematically while floating
point KL cancellation leaves disagreement near 1e-16. The query score's cube
root can magnify that residual to roughly 1e-6. A different argmax query then
changes later model state. During the review audit, near-identical initial
confidence was insufficient to ensure the same selected query across runtimes.
No rounding/stabilization or threshold retuning was introduced to conceal that
limitation. The name `js_disagreement` is a compatibility alias; the quantity is
mean KL to the learned weighted mixture, not generally Jensen-Shannon divergence.

## 6. Post-hoc paired anchor intervals and leakage identity

`src/analyze_paired_anchor_ci.py` computes fixed-stratum paired-seed bootstrap
intervals and a separate stratum-resampled sensitivity analysis for the Ask-only
mean and zero-shot/TDA tail contrasts. It uses 90 matched conditions per
comparison. These are post-hoc marginal intervals, not simultaneous intervals
or evidence that repeated stream orders are independent datasets.

The utility deliberately requires the existing confirmation/anchor/TDA result
CSVs and Ask-only summary JSONs at the relative paths named in its source. Its
five-percent provenance audit additionally requires the checksum-verified
historical post-hoc tar archive, extraction, and launcher. None of those data,
result files, event logs, or archives is bundled. Supply them separately before
running from the release root:

```bash
python src/analyze_paired_anchor_ci.py --output /work/analysis/paired_anchor_ci
```

The one-percent event-derived score-repair identity is checked rather than
extrapolated: inflation per run is actual query fraction times queried error
rate. Macro and pooled averages differ. Historical five-percent runs retained
only summaries, so their queried correctness and worst-window score repair
cannot be recovered. A new event-enabled replay would be a new experiment, not
recovery of those historical trajectories. No missing events are synthesized.

## 7. New full-event paired one/five-percent budget replay

`src/run_budget_leakage_review.py` runs 180 new post-hoc conditions: the six
caches from section 1, three streams, seeds 20-24, and exact 1%/5% budgets.
It needs NumPy and pandas (included in the local reference replay requirements).
`environment/requirements-budget-leakage.txt` separately records the observed
server versions for this new experiment; use a separate Python 3.10 environment
when targeting that runtime, not a forced downgrade of the local environment.
This dependency record is not a clean-install-tested or complete BLAS lock.
It does not download images or caches. Supply exactly the six `.npz` caches
listed in section 1; features, text prototypes, labels, domains, paths, and
class identities must match the intended cache contract. No cache, prediction,
event, summary, or result data is bundled in this archive.

Set the three BLAS thread variables to 1 before launch, as shown above. Use a
new output directory containing the exact path component `budget_leakage_replay`
or `round3_budget_review`; the runner rejects historical result destinations:

```bash
python src/run_budget_leakage_review.py --cache-root /work/cache \
  --output-dir /work/analysis/budget_leakage_replay --workers 3
```

The runner keeps the existing controller, thresholds, and numerical tie handling
unchanged. It writes full immutable-prediction events, summaries, audit receipts,
cache/source/config/runtime hashes, paired differences, and an aggregate report.
Both budgets share each stream order and raw zero-shot reference. It verifies
exact query counts, one query per segment, segment-close reveals, no Ask/Adapt
overlap, cache identities, event hashes, and the per-run score-repair identity.
These event checks do not supply unrecorded per-step prototype/weight snapshots.

Resume is accepted only with the same recorded cache, controller source,
configuration, and runtime contract. Completed traces are rechecked; incomplete
files are preserved separately. Do not mix partial outputs from different
runtimes. The frozen runner SHA-256 for this experiment is
`fbea67a205abbe15a821e7524e3071ac76a5cbfdf44cf89c77a4000357bac4d2`.

This is a new same-runtime post-hoc experiment, not recovery of the historical
summary-only five-percent trajectories and not a replacement for the original
five-seed confirmation. Its paired marginal intervals do not establish an
unseen-domain or multiplicity-adjusted confirmation claim. The code release
contains no benchmark-result claims from an incomplete execution.

For a separate local verification after transferring the outputs, use:

```bash
python src/audit_budget_leakage_replay.py --root /work/analysis/budget_leakage_replay
```

This verifier expects the six caches under the release-root-relative path
`remote_artifacts/20260814_multibench_v6/cache`; those files must be supplied
separately. It reads events, summaries, receipts, metadata, source, and caches,
then writes only `independent_audit.json` inside the supplied result directory.
To additionally verify a separately supplied transfer archive, append
`--archive /work/transferred-results.tgz --archive-sha256 EXPECTED_SERVER_SHA256`.
Without those arguments, archive verification is explicitly `NOT_REQUESTED`.
The event files, transfer archive, and resulting audit report are not bundled.

## Checking a reproduction

Record Python, NumPy, BLAS, PyTorch/CUDA, platform, thread settings, source/cache
hashes, exact commands, and hardware. Check complete condition keys and query
counts before summarizing. Use paired condition comparisons, and do not count
five order seeds on one dataset as five independent datasets. Report backend
differences, missing logs, and failed archival-default checks explicitly.
