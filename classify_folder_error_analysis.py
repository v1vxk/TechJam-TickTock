#!/usr/bin/env python
"""
classify_folder_error_analysis.py — score a labelled folder and produce every metric + calibration graph.

    python classify_folder_error_analysis.py eval_subset --labels eval_subset/labels.json --bundle ./aigc_export --out-dir eval_report
    python classify_folder_error_analysis.py eval_subset --labels eval_subset/labels.json --bundle ./aigc_export --predictions results.json   # reuse classify_folder.py output, no re-inference

Labels: labels.json from make_eval_subset.py ({"files": [...], "labels": [...]}), a CSV with columns file,label[,source],
        or a .npy / .json array aligned with the folder's images sorted by name.  0 = real, 1 = ai.

Outputs in --out-dir:
  predictions.csv     file, label, p_raw, p_platt, p_isotonic, cluster, source
  metrics.json        accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, Brier, ECE, confusion matrix — for raw / platt / isotonic
  reliability.png     the three reliability diagrams (raw / Platt / isotonic) with bin counts
  roc.png             ROC curve with AUC
  histograms.png      score distributions per class for each calibration
  confusion.png       confusion matrix at the chosen threshold (calibrated probability)
  by_source.csv       accuracy / AUC per source, if the labels carry sources
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


# Read ground truth in any of three formats: labels.json from make_eval_subset.py (files + labels + sources), a CSV with
# file,label[,source], or a bare 0/1 array (.npy / .json) aligned with the folder's images sorted by name.
def load_labels(path, folder):
    p = Path(path); imgs = sorted(q.name for q in folder.iterdir() if q.suffix.lower() in IMG_EXTS)
    if p.suffix == '.json':
        d = json.load(open(p))
        if isinstance(d, dict) and 'files' in d:
            return pd.DataFrame({'file': d['files'], 'label': [int(x) for x in d['labels']], 'source': d.get('sources', ['?'] * len(d['files']))})
        arr = d if isinstance(d, list) else d['labels']
    elif p.suffix == '.csv':
        df = pd.read_csv(p); df['label'] = df['label'].astype(int)
        if 'source' not in df: df['source'] = '?'
        return df[['file', 'label', 'source']]
    elif p.suffix == '.npy':
        arr = np.load(p)
    else:
        raise SystemExit(f'unsupported labels file: {p}')
    arr = [int(x) for x in arr]
    if len(arr) != len(imgs): raise SystemExit(f'array has {len(arr)} labels but the folder has {len(imgs)} images (arrays are aligned to images sorted by name)')
    return pd.DataFrame({'file': imgs, 'label': arr, 'source': '?'})


# Expected Calibration Error: bin predictions into 10 equal-width bins, and average |observed fake rate - mean predicted p|
# weighted by bin size. 0 = perfectly calibrated.
def ece_score(y, p, n_bins=10):
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1); e = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser(description='Metrics + calibration graphs for a labelled folder.')
    ap.add_argument('folder'); ap.add_argument('--labels', required=True); ap.add_argument('--bundle', default='aigc_export')
    ap.add_argument('--out-dir', default=None); ap.add_argument('--predictions', default=None, help='results.json from classify_folder.py to skip inference')
    ap.add_argument('--threshold', type=float, default=0.5); ap.add_argument('--fast', action='store_true'); ap.add_argument('--device', default=None)
    ap.add_argument('--features-cache', default=None, help='parquet of per-image feature rows (default <folder>/features_cache.parquet). Written on the first run, reused by later runs with ANY bundle whose features are a subset (e.g. the forensics-only bundle) — no re-inference')
    a = ap.parse_args()
    folder = Path(a.folder); out = Path(a.out_dir or (folder.parent / (folder.name + '_report'))); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from aigc_inference import Detector, _logit, level_image
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, balanced_accuracy_score, brier_score_loss, precision_score, recall_score, f1_score, confusion_matrix)
    from sklearn.calibration import calibration_curve

    # Ground truth first, then the model (heavy extractors load lazily on the first image).
    lab = load_labels(a.labels, folder)
    det = Detector(a.bundle, device=a.device, zed_stride=(4 if a.fast else None))

    # ---- predictions: reuse or compute ----
    # Three ways to get raw forest votes: (a) reuse a classify_folder.py JSON; (b) reuse the per-image feature cache written by an
    # earlier run — any bundle whose feature columns are a subset can then be scored in seconds; (c) compute everything now and write
    # that cache for next time.
    if a.predictions:
        res = json.load(open(a.predictions))['results']
        pred = pd.DataFrame([{'file': r['file'], 'p_raw': r.get('p_ai_raw'), 'cluster': r.get('cluster')} for r in res if 'error' not in r])
        print(f'reusing {len(pred)} predictions from {a.predictions}')
    else:
        try: from tqdm import tqdm
        except Exception: tqdm = lambda x, **k: x
        cache_p = Path(a.features_cache) if a.features_cache else folder / 'features_cache.parquet'
        feats = pd.read_parquet(cache_p).set_index('file') if cache_p.exists() else None
        if feats is not None and all(c in feats.columns for c in det.cols) and set(lab.file) <= set(feats.index):
            print(f'reusing cached features from {cache_p} ({len(feats)} images) — scoring with this bundle only')
            X = feats.loc[lab.file, det.cols].to_numpy(float)
            pred = pd.DataFrame({'file': lab.file.values, 'p_raw': det.rf.predict_proba(X)[:, 1], 'cluster': feats.loc[lab.file, 'cluster'].values})
        else:
            rows, cache_rows, t0 = [], [], time.time()
            for f in tqdm(lab.file.tolist(), desc='featurising + scoring'):
                try:
                    row, k, extra = det.featurize(level_image(str(folder / f), q=det.C['LEVEL_JPEG_Q'], max_side=det.C['MAX_SIDE']))
                    rows.append({'file': f, 'p_raw': float(det.rf.predict_proba(row.values)[:, 1][0]), 'cluster': k})
                    cache_rows.append({'file': f, 'cluster': k, **row.iloc[0].to_dict()})
                except Exception as e:
                    print('  failed:', f, repr(e)[:80])
            pred = pd.DataFrame(rows); print(f'scored {len(pred)} images in {(time.time() - t0) / 60:.1f} min')
            pd.DataFrame(cache_rows).to_parquet(cache_p, index=False); print(f'feature cache written to {cache_p} (re-run with another bundle to score instantly)')
    # Join predictions with labels, then derive the two calibrated probabilities from the raw vote using the calibrators that ship
    # in the bundle: Platt (logistic on the log-odds) and isotonic (monotone step function).
    df = lab.merge(pred, on='file', how='inner')
    if len(df) < len(lab): print(f'warning: {len(lab) - len(df)} labelled files have no prediction')
    p_raw = df.p_raw.to_numpy(float)
    df['p_platt'] = det.platt.predict_proba(_logit(p_raw)[:, None])[:, 1]
    df['p_isotonic'] = det.iso.predict(p_raw)
    df[['file', 'label', 'p_raw', 'p_platt', 'p_isotonic', 'cluster', 'source']].to_csv(out / 'predictions.csv', index=False)
    y = df.label.to_numpy(int)

    # ---- metrics ----
    # Metrics for each of the three probability versions at the chosen threshold: accuracy, balanced accuracy, precision / recall / F1
    # for the AI class, recall on reals, ROC-AUC (threshold-free), Brier, ECE and the confusion counts.
    metrics = {}
    for name in ('p_raw', 'p_platt', 'p_isotonic'):
        p = df[name].to_numpy(float); yhat = (p > a.threshold).astype(int)
        cm = confusion_matrix(y, yhat, labels=[0, 1])
        metrics[name[2:]] = dict(n=int(len(y)), threshold=a.threshold,
                                 accuracy=accuracy_score(y, yhat), balanced_accuracy=balanced_accuracy_score(y, yhat),
                                 precision_ai=precision_score(y, yhat, zero_division=0), recall_ai=recall_score(y, yhat, zero_division=0), f1_ai=f1_score(y, yhat, zero_division=0),
                                 real_recall=float(cm[0, 0] / max(1, cm[0].sum())), roc_auc=(roc_auc_score(y, p) if len(np.unique(y)) > 1 else None),
                                 brier=brier_score_loss(y, p), ece=ece_score(y, p), confusion=dict(tn=int(cm[0, 0]), fp=int(cm[0, 1]), fn=int(cm[1, 0]), tp=int(cm[1, 1])))
    json.dump(metrics, open(out / 'metrics.json', 'w'), indent=2)
    tab = pd.DataFrame(metrics).T.drop(columns=['confusion'])
    print('\n' + tab.round(4).to_string())

    # ---- reliability diagrams ----
    # Reliability diagrams: mean predicted probability per bin (x) vs observed fake rate (y), with the bin counts annotated.
    # A calibrated model sits on the diagonal.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, name in zip(axes, ('p_raw', 'p_platt', 'p_isotonic')):
        p = df[name].to_numpy(float); m = metrics[name[2:]]
        frac, mean_p = calibration_curve(y, p, n_bins=10, strategy='uniform'); counts = np.bincount(np.clip((p * 10).astype(int), 0, 9), minlength=10)
        ax.plot([0, 1], [0, 1], 'k--', lw=1); ax.plot(mean_p, frac, 'o-')
        for x_, y_, c_ in zip(mean_p, frac, counts[counts > 0]): ax.annotate(str(c_), (x_, y_), fontsize=7, textcoords='offset points', xytext=(3, 3))
        ax.set_title(f'{folder.name} — {name[2:]}\nAUC {m["roc_auc"]:.3f} · Brier {m["brier"]:.3f} · ECE {m["ece"]:.3f}', fontsize=9)
        ax.set_xlabel('predicted probability (ai)'); ax.set_ylabel('observed frequency of ai'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout(); plt.savefig(out / 'reliability.png', dpi=130); plt.close()

    # ---- ROC ----
    # ROC curve on the Platt probabilities.
    plt.figure(figsize=(4.8, 4.5))
    if len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, df.p_platt); plt.plot(fpr, tpr, label=f'AUC {roc_auc_score(y, df.p_platt):.3f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1); plt.xlabel('false positive rate (real called ai)'); plt.ylabel('true positive rate (ai caught)'); plt.legend(); plt.title(folder.name)
    plt.tight_layout(); plt.savefig(out / 'roc.png', dpi=130); plt.close()

    # ---- score histograms ----
    # Score histograms per class: how well the two classes separate, and where the threshold falls.
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for ax, name in zip(axes, ('p_raw', 'p_platt', 'p_isotonic')):
        ax.hist(df.loc[y == 0, name], bins=20, range=(0, 1), alpha=.6, label='real'); ax.hist(df.loc[y == 1, name], bins=20, range=(0, 1), alpha=.6, label='ai')
        ax.axvline(a.threshold, color='k', ls='--', lw=1); ax.set_title(name[2:]); ax.set_xlabel('p(ai)'); ax.legend()
    plt.tight_layout(); plt.savefig(out / 'histograms.png', dpi=130); plt.close()

    # ---- confusion matrix ----
    # Confusion matrix at the threshold (Platt probabilities).
    cm = metrics['platt']['confusion']; M = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
    plt.figure(figsize=(3.8, 3.4)); plt.imshow(M, cmap='Blues'); plt.xticks([0, 1], ['pred real', 'pred ai']); plt.yticks([0, 1], ['true real', 'true ai'])
    for i in range(2):
        for j in range(2): plt.text(j, i, str(M[i, j]), ha='center', va='center', color='white' if M[i, j] > M.max() / 2 else 'black', fontsize=12)
    plt.title(f'Platt, threshold {a.threshold}'); plt.tight_layout(); plt.savefig(out / 'confusion.png', dpi=130); plt.close()

    # ---- per source / per cluster ----
    # Per-source and per-cluster tables: accuracy, AUC (NaN when a group holds a single class), mean p(AI) and the AI share.
    def breakdown(by):
        rows = []
        for k, g in df.groupby(by):
            yy, pp = g.label.to_numpy(int), g.p_platt.to_numpy(float)
            rows.append({by: k, 'n': len(g), 'accuracy': accuracy_score(yy, pp > a.threshold), 'auc': roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else np.nan, 'mean_p_ai': pp.mean(), 'ai_share': yy.mean()})
        return pd.DataFrame(rows)
    if (df.source != '?').any():
        bs = breakdown('source'); bs.to_csv(out / 'by_source.csv', index=False); print('\n-- by source --\n' + bs.round(3).to_string(index=False))
    if df.cluster.notna().any():
        bc = breakdown('cluster'); bc.to_csv(out / 'by_cluster.csv', index=False); print('\n-- by cluster --\n' + bc.round(3).to_string(index=False))
    print(f'\nreport written to {out}')


if __name__ == '__main__':
    main()
