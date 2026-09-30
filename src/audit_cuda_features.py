"""Independent GPU precision audit; never replaces canonical event predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This audit requires an actual CUDA device')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    for path in sorted(args.cache_root.glob('*.npz')):
        with np.load(path, allow_pickle=False) as archive:
            x, text, labels = archive['features'], archive['text_features'], archive['labels']
        # Exact scalar-row expression from run_streams.py, rather than batched
        # BLAS argmax; the archival event trace remains the primary reference.
        cpu_pred = np.asarray([np.argmax(row @ text.T) for row in x])
        for dtype in (torch.float32, torch.float64):
            gx = torch.tensor(x, dtype=dtype, device='cuda')
            gt = torch.tensor(text, dtype=dtype, device='cuda')
            gx = torch.nn.functional.normalize(gx, dim=1)
            gt = torch.nn.functional.normalize(gt, dim=1)
            # Warm-up, then real synchronized timing (not end-to-end encoding).
            _ = gx @ gt.T
            torch.cuda.synchronize()
            started = time.perf_counter()
            logits = 100 * gx @ gt.T
            pred = logits.argmax(dim=1)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            pred = pred.cpu().numpy()
            result = dict(cache=path.stem, dtype=str(dtype), samples=len(labels),
                          online_accuracy=float(np.mean(pred == labels)),
                          canonical_cpu_accuracy=float(np.mean(cpu_pred == labels)),
                          prediction_disagreements=int(np.sum(pred != cpu_pred)),
                          synchronized_seconds=elapsed, cache_sha256=sha256(path))
            rows.append(result)
            print(json.dumps(result), flush=True)
            del gx, gt, logits
        torch.cuda.empty_cache()
    with (args.output_dir / 'cuda_feature_audit.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = dict(scope='full cached features; independent normalized FP32/FP64 GPU inference audit, not training',
                  device=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
                  numpy=np.__version__, platform=platform.platform(), tf32=False, rows=rows)
    (args.output_dir / 'cuda_feature_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
