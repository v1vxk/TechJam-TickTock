#!/usr/bin/env python
"""
forensics_only_forest.py — train and compare forests on different feature subsets, entirely from the cached feature table
(no images, no GPU, no re-downloading). Answers: how much of the score comes from CLIP (content) vs the forensic features?

    python forensics_only_forest.py aigc_rf_cache-XXXX.zip --out-dir forensics_report                                        # the cache ZIP as downloaded (nothing is extracted; only 2 members are read)
    python forensics_only_forest.py aigc_rf_cache-XXXX.zip --out-dir forensics_report --export-bundle aigc_export_forensics   # also write a drop-in bundle for classify_folder.py / dashboard.py
    python forensics_only_forest.py /path/to/aigc_rf_cache --grid                                                           # an unzipped cache folder works too; --grid re-tunes hyper-parameters per variant
    python forensics_only_forest.py aigc_rf_cache --export-bundle aigc_export_nozed --export-variant no-zed                 # fast-inference bundle (no SReC at scoring time)
    python forensics_only_forest.py aigc_rf_cache.zip --only no-zed --export-bundle aigc_export_nozed --export-variant no-zed   # train/export just one variant (seconds)

Variants (same train / calibration / test / demo splits as the notebook, so numbers are directly comparable):
  full                 every feature the notebook used  (physics + FFT + AEROBLADE + ZED + CLIP PCA + CLIP probe + cluster one-hot)
  no-zed               everything except ZED — a bundle exported from it skips the SReC coder at inference (the slowest step on CPU)
  no-zed-no-aero       everything except ZED and AEROBLADE — physics + FFT + all CLIP-derived features; inference runs only CLIP (~3-4 s per image on CPU)
  forensics+clusters   drops the CLIP PCA and the CLIP probe, keeps the cluster one-hot (clusters come from CLIP, so this still carries a content prior)
  forensics            physics + FFT + AEROBLADE + ZED only — no semantic information at all
  forensics-no-zed     physics + FFT + AEROBLADE only
  forensics+clusters-no-zed   physics + FFT + AEROBLADE + cluster one-hot — the practical middle ground: no CLIP columns in the forest, no ZED at inference
  physics+fft          the parameter-free features alone (what the detector can do with zero learned extractors)

Outputs in --out-dir: comparison.csv / comparison.png (AUC, accuracy, Brier, ECE on test and demo per variant), robustness.png
(accuracy vs transform level per variant, test + demo), by_generator.csv, by_cluster.csv, importance_<variant>.png, reliability_<variant>.png.
"""
import argparse, json, shutil, time, copy, warnings, io, zipfile
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold, ParameterGrid
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
import joblib

# Feature subsets to compare. Each entry is a predicate on a column name: keep the column if it returns True.
# Column prefixes: aniso_/xscale_/nsd_/res_corr_ (physics), fft_, aero_ (AEROBLADE), zed_ (ZED), clip_pc*/clip_probe (CLIP), cl_* (cluster one-hot).
VARIANTS = {
    'full':               lambda c: True,
    'no-zed':             lambda c: not c.startswith('zed_'),                                   # everything except the (slow) ZED coder features
    'no-zed-no-aero':     lambda c: not c.startswith(('zed_', 'aero_')),                        # physics + FFT + CLIP PCA + probe + clusters: no VAEs, no coder -> fastest (CLIP only) inference
    'forensics+clusters': lambda c: not c.startswith('clip_'),
    'forensics':          lambda c: not (c.startswith('clip_') or c.startswith('cl_')),
    'forensics-no-zed':   lambda c: not (c.startswith(('clip_', 'cl_', 'zed_'))),               # physics + FFT + AEROBLADE only
    'forensics+clusters-no-zed': lambda c: not (c.startswith(('clip_', 'zed_'))),               # physics + FFT + AEROBLADE + cluster one-hot (no CLIP columns, no ZED)
    'physics+fft':        lambda c: not (c.startswith(('clip_', 'cl_', 'aero_', 'zed_'))),
}
# Prefix -> family name, used to summarise feature importance per family.
FAMILIES = [('anisotropy', 'aniso_'), ('cross-scale', 'xscale_'), ('noise-signal', 'nsd_'), ('channel corr', 'res_corr_'), ('fft', 'fft_'),
            ('CLIP PCA', 'clip_pc'), ('CLIP probe', 'clip_probe'), ('AEROBLADE', 'aero_'), ('ZED', 'zed_'), ('cluster', 'cl_')]
family = lambda c: next((n for n, p in FAMILIES if c.startswith(p)), 'other')


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))

def ece(y, p, n_bins=10):
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1); e = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

def obj(s): return np.asarray(s.to_numpy(dtype=object))


# The training cache may be the Drive zip or an unzipped folder; read one member either way without extracting anything.
def read_cache_member(cache, member):
    """Bytes of features/... or aigc_export/... from either an unzipped cache folder or the cache zip (read in place, nothing extracted)."""
    cache = Path(cache)
    if cache.is_dir():
        return (cache / member).read_bytes()
    with zipfile.ZipFile(cache) as z:
        hits = [n for n in z.namelist() if n.replace('\\', '/').endswith('/' + member) or n == member]
        if not hits: raise SystemExit(f'{member} not found inside {cache.name}')
        return z.read(hits[0])


# Train one variant on the TRAIN split (same rows and seed for every variant), optionally re-tuning hyper-parameters with
# group-aware CV (a clean/augmented pair never straddles folds), then fit Platt and isotonic calibrators on the CALIBRATION split.
def fit_variant(t, cols, params, seed, grid, n_jobs):
    tr = t.split == 'train'; X, y, g = t.loc[tr, cols].to_numpy(float), t.loc[tr, 'label'].to_numpy(int), obj(t.loc[tr, 'uid'])
    if grid:
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g)); best, best_s = None, -1
        for p in ParameterGrid({'n_estimators': [400], 'max_depth': [None, 16], 'min_samples_leaf': [1, 4, 10], 'max_features': ['sqrt', 0.33]}):
            s = np.mean([roc_auc_score(y[b], RandomForestClassifier(**p, class_weight='balanced', n_jobs=n_jobs, random_state=seed).fit(X[a], y[a]).predict_proba(X[b])[:, 1]) for a, b in folds])
            if s > best_s: best, best_s = p, s
        params = best; print(f'    grid → {params} (cv AUC {best_s:.4f})')
    rf = RandomForestClassifier(**params, class_weight='balanced', oob_score=True, n_jobs=n_jobs, random_state=seed).fit(X, y)
    ca = t.split == 'calib'; pc = rf.predict_proba(t.loc[ca, cols].to_numpy(float))[:, 1]; yc = t.loc[ca, 'label'].to_numpy(int)
    platt = LogisticRegression(C=1e6, max_iter=2000).fit(_logit(pc)[:, None], yc); iso = IsotonicRegression(out_of_bounds='clip').fit(pc, yc)
    return rf, platt, iso, params


# Raw forest vote, Platt probability and isotonic probability for a feature matrix.
def predict(rf, platt, iso, X):
    p = rf.predict_proba(X)[:, 1]; return p, platt.predict_proba(_logit(p)[:, None])[:, 1], iso.predict(p)


def main():
    ap = argparse.ArgumentParser(description='Full vs forensics-only forests from the cached feature table.')
    ap.add_argument('cache', help='the aigc_rf_cache ZIP as downloaded from Drive, or the unzipped folder (needs features/table_main.parquet and aigc_export/meta.json)')
    ap.add_argument('--base-bundle', default=str(Path(__file__).resolve().parent), help='folder with the full model files (meta.json, *.joblib, weights) used as the template for --export-bundle (default: the folder this script is in)')
    ap.add_argument('--out-dir', default='forensics_report'); ap.add_argument('--grid', action='store_true', help='re-tune hyper-parameters per variant (slower)')
    ap.add_argument('--export-bundle', default=None, help='write a drop-in bundle for the forensics variant (copies aigc_export and swaps the forest/calibrators/feature list)')
    ap.add_argument('--export-variant', default='forensics', choices=list(VARIANTS)); ap.add_argument('--perm-repeats', type=int, default=5); ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--only', nargs='+', choices=list(VARIANTS), default=None, help='train only these variants (e.g. --only no-zed forensics); default: all seven for the full comparison')
    a = ap.parse_args()
    cache = Path(a.cache); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(read_cache_member(cache, 'aigc_export/meta.json')); cols_all = meta['feature_cols']; seed = int(meta['config']['SEED'])
    # The notebook's assembled feature table: one row per (image, clean/augmented), with its split assignment, cluster, dataset,
    # generator and transform columns alongside the features. Splits are reused as-is so numbers are comparable with Colab's.
    t = pd.read_parquet(io.BytesIO(read_cache_member(cache, 'features/table_main.parquet')))
    assert all(c in t.columns for c in cols_all), 'feature table does not match meta.json feature list'
    t[cols_all] = t[cols_all].replace([np.inf, -np.inf], np.nan)
    t[cols_all] = t[cols_all].fillna(t.loc[t.split == 'train', cols_all].median()).fillna(0.0)
    has_demo = (t.split == 'demo').any()
    print(f'table: {len(t)} rows | splits {t.split.value_counts().to_dict()} | {len(cols_all)} features | notebook params {meta["best_params"]}')

    rows, preds, models = [], {}, {}
    selected = list(a.only) if a.only else list(VARIANTS)
    if a.export_bundle and a.export_variant not in selected: selected.append(a.export_variant)
    # Train + evaluate every selected variant on test (and demo, if present); collect metrics for the three calibrations.
    for name in selected:
        keep = VARIANTS[name]; cols = [c for c in cols_all if keep(c)]
        if not cols: continue
        t0 = time.time(); print(f'\n[{name}] {len(cols)} features: ' + ', '.join(f'{k}={v}' for k, v in pd.Series([family(c) for c in cols]).value_counts().items()))
        rf, platt, iso, params = fit_variant(t, cols, dict(meta['best_params']), seed, a.grid, a.n_jobs)
        models[name] = (rf, platt, iso, cols, params)
        for split in (['test', 'demo'] if has_demo else ['test']):
            m = t.split == split; y = t.loc[m, 'label'].to_numpy(int); p_raw, p_platt, p_iso = predict(rf, platt, iso, t.loc[m, cols].to_numpy(float))
            preds[(name, split)] = p_platt
            for cal, p in (('raw', p_raw), ('platt', p_platt), ('isotonic', p_iso)):
                rows.append(dict(variant=name, split=split, calibration=cal, n=int(m.sum()), auc=roc_auc_score(y, p), accuracy=accuracy_score(y, p > 0.5),
                                 balanced_accuracy=balanced_accuracy_score(y, p > 0.5), brier=brier_score_loss(y, p), ece=ece(y, p)))
        print(f'    OOB acc {rf.oob_score_:.4f} | ' + ' | '.join(f'{s}: AUC {r["auc"]:.4f} acc {r["accuracy"]:.4f}' for s in (['test', 'demo'] if has_demo else ['test'])
                                                          for r in rows if r['variant'] == name and r['split'] == s and r['calibration'] == 'platt') + f'  ({time.time()-t0:.0f}s)')
    # Comparison table (Platt) across variants and splits.
    res = pd.DataFrame(rows); res.to_csv(out / 'comparison.csv', index=False)
    pl = res[res.calibration == 'platt'].pivot(index='variant', columns='split', values=['auc', 'accuracy', 'brier', 'ece']).reindex([v for v in VARIANTS if v in models])
    print('\n===== Platt-calibrated, by variant =====\n' + pl.round(4).to_string())

    # ---- comparison figure ----
    splits = ['test', 'demo'] if has_demo else ['test']
    # Bar chart of AUC / accuracy per variant and split.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4)); x = np.arange(len(pl)); w = 0.8 / len(splits)
    for ax, metric in zip(axes, ('auc', 'accuracy')):
        for i, s in enumerate(splits):
            v = pl[(metric, s)].values; ax.bar(x + i * w, v, w, label=s)
            for xi, vi in zip(x + i * w, v): ax.text(xi, vi + 0.002, f'{vi:.3f}', ha='center', fontsize=7)
        ax.set_xticks(x + w * (len(splits) - 1) / 2); ax.set_xticklabels(pl.index, rotation=15); ax.set_ylim(max(0.5, pl[metric].values.min() - 0.05), 1.0); ax.set_title(metric + ' (Platt)'); ax.legend()
    plt.tight_layout(); plt.savefig(out / 'comparison.png', dpi=130); plt.close()

    # ---- robustness: accuracy vs transform level per variant ----
    # Robustness: accuracy per transform × level for each variant (the 'aug' column of the table tells us what each row had applied).
    fig, axes = plt.subplots(1, len(splits), figsize=(6.5 * len(splits), 4.2), squeeze=False)
    for ax, s in zip(axes[0], splits):
        d = t[t.split == s].copy()
        for name in models:
            d['ok'] = ((preds[(name, s)] > 0.5) == (d.label.values == 1))
            g = d[d.aug_type != 'none'].groupby(['aug_type', 'aug_level']).ok.mean().reset_index()
            g['x'] = g.aug_type + ' ' + g.aug_level.astype(str)
            ax.plot(range(len(g)), g.ok.values, 'o-', label=f'{name} (clean {d[d.aug_type=="none"].ok.mean():.3f})', ms=4)
        ax.set_xticks(range(len(g))); ax.set_xticklabels(g.x, rotation=70, fontsize=7); ax.set_ylabel('accuracy'); ax.set_title(f'{s}: accuracy per transform × level'); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(out / 'robustness.png', dpi=130); plt.close()

    # ---- per generator / per cluster (Platt, each variant) ----
    # Accuracy / AUC per generator and per cluster, per variant.
    for by in ('generator', 'cluster'):
        tabs = []
        for s in splits:
            d = t[t.split == s]
            for name in models:
                d2 = d.assign(p=preds[(name, s)])
                for k, g in d2.groupby(by):
                    yy = g.label.to_numpy(int)
                    tabs.append({by: k, 'split': s, 'variant': name, 'n': len(g), 'accuracy': accuracy_score(yy, g.p > 0.5), 'auc': roc_auc_score(yy, g.p) if len(np.unique(yy)) > 1 else np.nan})
        pd.DataFrame(tabs).to_csv(out / f'by_{by}.csv', index=False)
    bg = pd.read_csv(out / 'by_generator.csv'); bg = bg[bg.split == splits[-1]].pivot(index='generator', columns='variant', values='accuracy')
    print(f'\n-- accuracy by generator on {splits[-1]} (Platt) --\n' + bg.round(3).to_string())

    # ---- importance + reliability per variant ----
    # Per variant: permutation importance on the test split (summed by family), and a reliability diagram per split.
    te = t.split == 'test'; y_te = t.loc[te, 'label'].to_numpy(int)
    for name, (rf, platt, iso, cols, params) in models.items():
        pi = permutation_importance(rf, t.loc[te, cols].to_numpy(float), y_te, scoring='accuracy', n_repeats=a.perm_repeats, n_jobs=a.n_jobs, random_state=seed)
        imp = pd.DataFrame({'feature': cols, 'family': [family(c) for c in cols], 'perm_acc': pi.importances_mean, 'std': pi.importances_std}).sort_values('perm_acc', ascending=False)
        imp.to_csv(out / f'importance_{name}.csv', index=False); top = imp.head(25)
        plt.figure(figsize=(7, 7)); plt.barh(top.feature[::-1], top.perm_acc[::-1], xerr=top['std'][::-1]); plt.title(f'{name}: permutation importance (test, mean decrease in accuracy)'); plt.tight_layout(); plt.savefig(out / f'importance_{name}.png', dpi=130); plt.close()
        fam = imp.groupby('family').perm_acc.sum().sort_values(ascending=False); print(f'\n[{name}] importance by family: ' + ', '.join(f'{k} {v:.3f}' for k, v in fam.items()))
        fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 4.2), squeeze=False)
        for ax, s in zip(axes[0], splits):
            m = t.split == s; yy = t.loc[m, 'label'].to_numpy(int); p = preds[(name, s)]
            frac, mp = calibration_curve(yy, p, n_bins=10); ax.plot([0, 1], [0, 1], 'k--', lw=1); ax.plot(mp, frac, 'o-')
            ax.set_title(f'{name} / {s} (Platt)\nAUC {roc_auc_score(yy, p):.3f} · ECE {ece(yy, p):.3f}', fontsize=9); ax.set_xlabel('predicted p(ai)'); ax.set_ylabel('observed'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        plt.tight_layout(); plt.savefig(out / f'reliability_{name}.png', dpi=130); plt.close()

    # ---- optional: drop-in bundle for the chosen variant ----
    # Optional: write a complete drop-in bundle for one variant — copy the model files from the base bundle (PCAs, cluster model,
    # weights), replace the forest / calibrators / SHAP sub-forest, and rewrite meta.json with the reduced feature list and new medians.
    if a.export_bundle:
        rf, platt, iso, cols, params = models[a.export_variant]; dst = Path(a.export_bundle); base = Path(a.base_bundle)
        if not (base / 'meta.json').exists(): raise SystemExit(f'--base-bundle {base} has no meta.json (point it at the folder holding the full model files)')
        if dst.exists(): shutil.rmtree(dst)
        dst.mkdir(parents=True); f = meta['files']
        for fname in set(f.values()):                                   # copy only the model files that exist (weights, PCAs, cluster model, …); the forest/calibrators are replaced below
            if (base / fname).exists(): shutil.copy(base / fname, dst / fname)
        joblib.dump(rf, dst / f['rf']); joblib.dump(platt, dst / f['platt']); joblib.dump(iso, dst / f['isotonic'])
        sub = copy.deepcopy(rf); sub.estimators_ = rf.estimators_[:int(meta['config'].get('SHAP_MAX_TREES', 100))]; sub.n_estimators = len(sub.estimators_); joblib.dump(sub, dst / f['shap_forest'])
        m2 = dict(meta); m2['feature_cols'] = cols; m2['best_params'] = params; m2['variant'] = a.export_variant
        m2['train_medians'] = {c: float(v) for c, v in t.loc[t.split == 'train', cols].median().items()}
        m2['test_metrics'] = {'platt': res[(res.variant == a.export_variant) & (res.split == 'test') & (res.calibration == 'platt')].iloc[0][['auc', 'accuracy', 'balanced_accuracy', 'brier', 'ece']].to_dict()}
        json.dump(m2, open(dst / 'meta.json', 'w'), indent=2, default=str)
        print(f'\nexported {a.export_variant} bundle → {dst}  (use with:  classify_folder.py … --bundle {dst})')
    print(f'\nreport written to {out}')


if __name__ == '__main__':
    main()
