"""Independent FP64 CUDA TDA replay, batching stream conditions not time steps.

Cache selection stays chronological. No oracle labels enter the adapter. This
is a cross-backend audit, not a substitute for the canonical NumPy matrix.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from run_tda_review_matrix import stream_order, ece, worst500, hash_file, run_replay, OFFICIAL_COMMIT


def replay(cache, conditions, verbose=True):
    device = 'cuda'
    dtype = torch.float64
    orders = np.stack([stream_order(cache['labels'], cache['domains'], st, sd) for st, sd in conditions])
    order = torch.tensor(orders, device=device)
    x = torch.tensor(cache['features'], dtype=dtype, device=device)
    text = torch.tensor(cache['text_features'], dtype=dtype, device=device)
    base_all = 100 * x @ text.T
    count, length = orders.shape
    classes, dim = text.shape
    batch = torch.arange(count, device=device)
    pos = torch.zeros((count, classes, 3, dim), dtype=dtype, device=device)
    neg = torch.zeros((count, classes, 2, dim), dtype=dtype, device=device)
    ploss = torch.full((count, classes, 3), float('inf'), dtype=dtype, device=device)
    nloss = torch.full((count, classes, 2), float('inf'), dtype=dtype, device=device)
    nmask = torch.zeros((count, classes, 2, classes), dtype=dtype, device=device)
    prediction = torch.empty((count, length), dtype=torch.int64, device=device)
    confidence = torch.empty((count, length), dtype=dtype, device=device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for t in range(length):
        feature = x[order[:, t]]
        base = base_all[order[:, t]]
        probs = torch.softmax(base, dim=1)
        label = base.argmax(dim=1)  # model pseudo-label, NOT a ground-truth label
        loss = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1) / np.log2(classes)
        slot = ploss[batch, label].argmax(dim=1)
        accept = loss < ploss[batch, label, slot]
        pos[batch, label, slot] = torch.where(accept[:, None], feature, pos[batch, label, slot])
        ploss[batch, label, slot] = torch.where(accept, loss, ploss[batch, label, slot])
        slot = nloss[batch, label].argmax(dim=1)
        accept = (loss > .2) & (loss < .5) & (loss < nloss[batch, label, slot])
        neg[batch, label, slot] = torch.where(accept[:, None], feature, neg[batch, label, slot])
        nloss[batch, label, slot] = torch.where(accept, loss, nloss[batch, label, slot])
        mask = ((probs > .03) & (probs < 1.0)).to(dtype)
        nmask[batch, label, slot] = torch.where(accept[:, None], mask, nmask[batch, label, slot])
        paff = torch.einsum('bckd,bd->bck', pos, feature)
        pw = torch.exp(-5 * (1 - paff)) * torch.isfinite(ploss)
        naff = torch.einsum('bckd,bd->bck', neg, feature)
        nw = torch.exp(-(1 - naff)) * torch.isfinite(nloss)
        adapted = base + 2 * pw.sum(dim=2) - .117 * (nw[..., None] * nmask).sum(dim=(1, 2))
        final = torch.softmax(adapted, dim=1)
        prediction[:, t] = final.argmax(dim=1)
        confidence[:, t] = final.max(dim=1).values
        if verbose and (t + 1) % 5000 == 0:
            torch.cuda.synchronize()
            print(f'GPU completed {t+1}/{length} samples in each of {count} streams', flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return orders, prediction.cpu().numpy(), confidence.cpu().numpy(), elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[10, 11, 12, 13, 14])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conditions = [(st, sd) for st in ('iid_random', 'abrupt_domain', 'class_correlated') for sd in args.seeds]
    rows = []
    smoke = None
    for path in sorted(args.cache_root.glob('*.npz')):
        with np.load(path, allow_pickle=False) as z:
            cache = {k: z[k] for k in ('features', 'text_features', 'labels', 'domains')}
        if smoke is None:
            small = {k: (v if k == 'text_features' else v[:256]) for k, v in cache.items()}
            so, sp, sc, _ = replay(small, [('iid_random', 10)], verbose=False)
            _, ref = run_replay(small, 'iid_random', 10, 'official_log2')
            smoke = dict(samples=256, prediction_disagreements=int(np.sum(sp[0] != ref['prediction'])),
                         confidence_max_abs_error=float(np.max(np.abs(sc[0]-ref['confidence']))))
            if smoke['prediction_disagreements']:
                raise RuntimeError(f'CUDA smoke disagreement: {smoke}')
            print('CUDA versus NumPy smoke:', smoke, flush=True)
        order, pred, conf, elapsed = replay(cache, conditions)
        # Target labels are consumed only after all predictions are finalized.
        correctness = (pred == cache['labels'][order]).astype(np.float64)
        for i, (st, sd) in enumerate(conditions):
            row = dict(cache=path.stem, stream=st, seed=sd, samples=len(pred[i]), strategy='TDA_CUDA_FP64',
                       online_accuracy=float(correctness[i].mean()), worst_window_accuracy=worst500(correctness[i]),
                       expected_calibration_error=ece(conf[i], correctness[i]), query_count=0,
                       cache_sha256=hash_file(path), batch_elapsed_seconds=elapsed)
            rows.append(row)
            np.savez_compressed(args.output_dir / f'{path.stem}__{st}__seed{sd}.npz',
                                original_index=order[i], prediction=pred[i], confidence=conf[i])
        with (args.output_dir / 'tda_cuda_runs.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        print(path.stem, 'completed CUDA batch', elapsed, 'seconds', flush=True)
        (args.output_dir / 'manifest.json').write_text(json.dumps(dict(
            status='PASS' if len(rows) == 6*len(conditions) else 'RUNNING', device=torch.cuda.get_device_name(0),
            torch=torch.__version__, cuda=torch.version.cuda, dtype='float64', tf32=False,
            official_commit=OFFICIAL_COMMIT, runs=len(rows), expected_runs=6*len(conditions),
            validation=smoke, scope='independent batched-stream cross-backend audit; not training'), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
