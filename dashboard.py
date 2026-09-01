"""
dashboard.py — Streamlit front-end for the exported AI-image detector(s).

    pip install streamlit
    python -m streamlit run dashboard.py                     # discovers every model folder next to this script
    python -m streamlit run dashboard.py -- --bundle path    # add one more bundle folder to the list

What the page does
------------------
1. Sidebar → pick the PRIMARY model. Every folder containing a meta.json (this folder = the full model, aigc_export_* = the
   variants exported by forensics_only_forest.py) is listed with the feature families it uses.
2. Sidebar → optionally tick other models to COMPARE: the same image is scored by all of them. Features are computed once with the
   model that has the largest feature set and reused for the others (that is why comparing is nearly free); a model whose features
   are not a subset falls back to computing its own.
3. Sidebar → "fast ZED": compute the ZED entropy map on a 4× pixel stride. Several-fold faster on CPU, slightly approximate; only
   affects models that use ZED (the no-zed bundles never run the coder at all).
4. Sidebar → chain any of the six training transforms (JPEG, blur, resize, noise, colour jitter, centre crop) at the training levels
   or a custom value. They are applied in the order selected, to the levelled image, exactly as in training.
5. Upload an image → the clean and transformed versions are shown side by side → "classify" scores both with every selected model,
   showing the calibrated p(AI), the verdict, the CLIP cluster, and (for the primary model) the top SHAP feature contributions.

Notes
-----
* p(AI) is the Platt-calibrated probability; verdict = AI if p(AI) > 0.5.
* The first classification loads CLIP / VAEs / SReC (a few minutes on the first ever run while weights download); later ones are
  ~10–60 s on CPU depending on the model, ~1 s on a GPU.
* Models are cached for the session with st.cache_resource, so switching between them is instant after the first load.
"""
import sys, io, time, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aigc_inference import Detector, apply_aug, level_image, AUG_HELP, _logit


# --------------------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------------------
def _args():
    """Streamlit passes everything after '--' to the script; we only understand --bundle and --fast."""
    ap = argparse.ArgumentParser(); ap.add_argument('--bundle', default=None); ap.add_argument('--fast', action='store_true')
    a, _ = ap.parse_known_args(); return a


def _show(img):
    """st.image API changed across versions (use_container_width -> width='stretch'); support both."""
    try: st.image(img, width='stretch')
    except TypeError: st.image(img, use_container_width=True)


def discover_bundles(extra=None):
    """Every folder (this one + immediate sub-folders + --bundle) that holds a meta.json is a usable model bundle.
    Returns {label: path}. The label summarises which feature families the forest uses so the user knows what each model is."""
    cands = [HERE] + sorted(p for p in HERE.iterdir() if p.is_dir()) + ([Path(extra)] if extra else [])
    out = {}
    for p in cands:
        if not (p / 'meta.json').exists(): continue
        try: meta = json.load(open(p / 'meta.json'))
        except Exception: continue
        cols = meta.get('feature_cols', []); variant = meta.get('variant', 'full' if p == HERE else p.name)
        fams = []
        if any(c.startswith(('aniso_', 'xscale_', 'nsd_', 'fft_')) for c in cols): fams.append('physics+fft')
        if any(c.startswith('aero_') for c in cols): fams.append('AEROBLADE')
        if any(c.startswith('zed_') for c in cols): fams.append('ZED')
        if any(c.startswith('clip_') for c in cols): fams.append('CLIP')
        if any(c.startswith('cl_') for c in cols): fams.append('clusters')
        label = f"{variant}  [{', '.join(fams)}]  ({p.name if p != HERE else 'this folder'})"
        out[label] = str(p)
    return out


@st.cache_resource(show_spinner='loading model …')
def load_detector(path, fast):
    """One Detector per (bundle, fast-flag); cached for the whole session so switching models is instant after the first load."""
    return Detector(path, zed_stride=(4 if fast else None), quiet=True)


def score_with_models(img, primary, others, explain):
    """Score one (levelled) image with the primary model and any comparison models.
    Features are computed once with the model that has the most columns; every model whose columns are a subset reuses them."""
    dets = [primary] + [d for d in others if d is not primary]
    base = max(dets, key=lambda d: len(d.cols))                    # the widest feature set available
    row, k, extra = base.featurize(img)                            # HxWx3 uint8 (already levelled) -> feature row + cluster id
    results = []
    for d in dets:
        if set(d.cols) <= set(base.cols):                          # reuse: select this model's columns from the shared row
            r = row.reindex(columns=d.cols)
            p_raw = float(d.rf.predict_proba(r.values)[:, 1][0]); p_cal = float(d.platt.predict_proba(_logit(np.array([p_raw]))[:, None])[0, 1])
            res = dict(p_ai=p_cal, p_ai_raw=p_raw, cluster=k, reused=True)
            if explain and d is primary and d.shap_forest is not None: res['top_contributions'] = d.explain(r)
        else:                                                      # rare: this model needs columns the base did not compute
            res = d.predict(img, explain=(explain and d is primary), already_levelled=True); res['reused'] = False
        res['verdict'] = 'ai' if res['p_ai'] > 0.5 else 'real'; results.append(res)
    return results


# --------------------------------------------------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------------------------------------------------
ARGS = _args()
st.set_page_config(page_title='AI-image detector', layout='wide')
st.title('AI-generated image detector')

bundles = discover_bundles(ARGS.bundle)
if not bundles:
    st.error('no model bundle found: this folder (or a sub-folder) must contain meta.json and the model files'); st.stop()

with st.sidebar:
    # ---- model choice -------------------------------------------------------------------------------------------------
    st.header('Models')
    labels = list(bundles.keys())
    default_i = next((i for i, l in enumerate(labels) if 'this folder' in l), 0)
    primary_label = st.selectbox('primary model', labels, index=default_i, help='the model whose verdict and SHAP explanation are shown in detail')
    compare_labels = st.multiselect('also compare with', [l for l in labels if l != primary_label], default=[],
                                    help='score the same image with these models too (features are shared, so this is cheap)')
    fast = st.checkbox('fast ZED (4× pixel stride, approximate)', value=ARGS.fast,
                       help='only affects models that use ZED; no-zed bundles never run the coder')
    try:
        det = load_detector(bundles[primary_label], fast)
        others = [load_detector(bundles[l], fast) for l in compare_labels]
    except Exception as e:
        st.error(f'could not load a model: {e}'); st.stop()
    st.caption(f"{len(det.cols)} features · K={det.K} clusters · ZED={det.zed_backend if det.use_zed else 'not used'} · "
               f"VAEs={', '.join(det.vaes_used) if det.use_aero else 'not used'} · {det.device}")
    if det.meta.get('smoke_test'): st.warning('this bundle was trained in SMOKE_TEST mode (synthetic data, mock CLIP/VAE) — pipeline testing only')
    tm = (det.meta.get('test_metrics') or {}).get('platt') or {}
    if tm: st.caption(f"held-out test: AUC {tm.get('auc', float('nan')):.3f} · accuracy {tm.get('acc', tm.get('accuracy', float('nan'))):.3f} · ECE {tm.get('ece', float('nan')):.3f}")

    # ---- transform chain ----------------------------------------------------------------------------------------------
    st.header('Transforms (applied in the order selected)')
    levels = det.transform_levels()                                # the organisers' table, straight from the training config
    chain = st.multiselect('add transforms', list(levels.keys()), default=[], help='the same six transforms used during training')
    specs = []
    for t in chain:
        opts = [str(v) for v in levels[t]] + ['custom']
        c1, c2 = st.columns([2, 1])
        with c1: choice = st.selectbox(f'{t} level', opts, key=f'lvl_{t}', help=AUG_HELP[t])
        with c2: seed = st.number_input('seed', 0, 10_000, 0, key=f'seed_{t}') if t in ('noise', 'jitter') else 0   # only the random transforms take a seed
        level = st.number_input(f'{t} custom value', value=float(levels[t][0]), key=f'custom_{t}', format='%.3f') if choice == 'custom' else float(choice)
        specs.append((t, level, int(seed)))
    explain = st.checkbox('show SHAP contributions (primary model)', value=True)

# ---- image ------------------------------------------------------------------------------------------------------------
up = st.file_uploader('image', type=['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tif', 'tiff'])
if up is None:
    st.info('upload an image to start'); st.stop()

raw = Image.open(io.BytesIO(up.getvalue()))
clean = level_image(raw, q=det.C['LEVEL_JPEG_Q'], max_side=det.C['MAX_SIDE'])   # exactly the training-time levelling (EXIF, RGB, ≤1024, JPEG q95)
img, applied = clean, []
for t, level, seed in specs:                                       # transforms are applied to the levelled image, as in training
    img, params = apply_aug(img, t, level, seed); applied.append({'transform': t, 'level': level, **params})

c1, c2 = st.columns(2)
with c1:
    st.subheader('clean (levelled)'); _show(clean); st.caption(f'{clean.shape[1]}×{clean.shape[0]}')
with c2:
    st.subheader('transformed' if applied else 'transformed (none selected)'); _show(img)
    if applied: st.json(applied, expanded=False)

# ---- classify ---------------------------------------------------------------------------------------------------------
if st.button('classify', type='primary'):
    with st.spinner('computing features …'):
        t0 = time.time()
        res_clean = score_with_models(clean, det, others, explain)
        res_aug = score_with_models(img, det, others, explain) if applied else None
        dt = time.time() - t0
    names = [primary_label.split('  [')[0]] + [l.split('  [')[0] for l in compare_labels]

    # primary verdict(s), big
    cols = st.columns(2 if res_aug else 1)
    for col, name, res in zip(cols, ['clean', 'transformed'], [res_clean, res_aug]):
        if res is None: continue
        r = res[0]
        with col:
            verdict = 'AI-GENERATED' if r['verdict'] == 'ai' else 'REAL'
            (st.error if r['verdict'] == 'ai' else st.success)(f'{name}: **{verdict}**  ·  p(AI) = {r["p_ai"]:.3f}   ({names[0]})')
            st.progress(float(r['p_ai']), text=f'calibrated p(AI) {r["p_ai"]:.3f}   (raw forest vote {r["p_ai_raw"]:.3f})')
            st.caption(f"cluster {r['cluster']}")
            if explain and r.get('top_contributions'):
                s = pd.Series(r['top_contributions']).iloc[::-1]
                st.bar_chart(s, horizontal=True, height=300)
                st.caption('SHAP contribution to p(AI): positive pushes towards AI, negative towards real')

    # comparison table across models (only when comparison models were selected)
    if others:
        st.subheader('model comparison')
        rows = []
        for i, name in enumerate(names):
            rows.append({'model': name, 'clean p(AI)': res_clean[i]['p_ai'], 'clean verdict': res_clean[i]['verdict'],
                         **({'transformed p(AI)': res_aug[i]['p_ai'], 'transformed verdict': res_aug[i]['verdict']} if res_aug else {})})
        st.dataframe(pd.DataFrame(rows).set_index('model').style.format({c: '{:.3f}' for c in rows[0] if 'p(AI)' in c}))
        st.caption('features were computed once with the widest model and reused; a model is re-featurised only if it needs columns the others lack')
    st.caption(f'{dt:.1f} s on {det.device}')