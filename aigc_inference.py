"""
aigc_inference.py — standalone inference library for the AI-generated-image detector.

Rebuilds the full pipeline (levelling -> native-res crop + CLIP view -> physics/FFT/AEROBLADE/ZED/CLIP features
-> unsupervised cluster one-hot -> random forest -> Platt calibration) from the folder written by the notebook's
export cell. The feature functions below are copied VERBATIM from the training notebook (generated file: do not edit
them here; regenerate from the notebook sources if the notebook changes).

    from aigc_inference import Detector
    d = Detector('aigc_export')
    d.predict('photo.jpg')          # -> {'verdict': 'real'|'ai', 'p_ai': 0.93, 'p_ai_raw': ..., 'cluster': 3, ...}

Requirements: numpy pandas scipy scikit-learn joblib opencv-python pillow torch transformers diffusers lpips shap
(torch/transformers/diffusers/lpips only needed when the bundle uses CLIP/AEROBLADE/ZED — i.e. always for the main tier).
"""
import io, os, sys, json, math, subprocess, warnings
from pathlib import Path
warnings.filterwarnings('ignore', category=RuntimeWarning)   # flat crops (e.g. screenshots) yield NaN physics features by design; they are imputed with training medians
import numpy as np, pandas as pd, joblib, cv2
from PIL import Image, ImageOps, ImageEnhance
from scipy import ndimage, stats
try:
    import torch
except Exception:
    torch = None

LOG_SCALE_MIN = -7.0

# ---------------------------------------------------------------------------------------------------------------------
#  Verbatim from the training notebook: augmentations, views, physics + FFT features, ZED-lite
# ---------------------------------------------------------------------------------------------------------------------
# --- the six training transforms (organisers' table). Each takes the uint8 image, a level and a seeded RNG,
# and returns (new image, dict of the exact parameters used) so the level can be recorded for error analysis.
# JPEG re-encode at quality q: round-trip through PIL's encoder and decode again.
def aug_jpeg(img, q, rng):
    buf = io.BytesIO(); Image.fromarray(img).save(buf, 'JPEG', quality=int(q)); return np.array(Image.open(io.BytesIO(buf.getvalue())).convert('RGB')), {'q': int(q)}

# Gaussian blur with sigma s pixels (kernel size derived from sigma by OpenCV).
def aug_blur(img, s, rng):
    return cv2.GaussianBlur(img, (0, 0), sigmaX=float(s), sigmaY=float(s)), {'sigma': float(s)}

# Resize down by factor s (area interpolation) and back up to the original size (cubic) — mimics a CDN thumbnail round-trip.
def aug_resize(img, s, rng):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(2, int(round(w * s))), max(2, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), {'scale': float(s), 'down': 'area', 'up': 'cubic'}

# Additive Gaussian noise with sigma s in [0,1] units, then clip and re-quantise to uint8.
def aug_noise(img, s, rng):
    x = img.astype(np.float32) / 255 + rng.normal(0, s, img.shape).astype(np.float32)
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8), {'sigma': float(s)}

# Colour jitter: brightness, contrast and saturation each scaled by a random factor in [1-a, 1+a]; the three factors are recorded.
def aug_jitter(img, a, rng):
    fb, fc, fs = rng.uniform(1 - a, 1 + a, 3)
    pil = Image.fromarray(img)
    pil = ImageEnhance.Brightness(pil).enhance(fb); pil = ImageEnhance.Contrast(pil).enhance(fc); pil = ImageEnhance.Color(pil).enhance(fs)
    return np.array(pil), {'brightness': round(float(fb), 3), 'contrast': round(float(fc), 3), 'saturation': round(float(fs), 3)}

# Centre crop keeping fraction f of each side (f=0.8 keeps 64% of the area).
def aug_crop(img, f, rng):
    h, w = img.shape[:2]; ch, cw = max(8, int(h * f)), max(8, int(w * f)); y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw].copy(), {'frac': float(f)}

# Top-left corners for n crops: the centre if n == 1, otherwise a roughly square grid spread over the image.
def crop_positions(h, w, crop, n):
    if n == 1: return [((h - crop) // 2, (w - crop) // 2)]
    g = int(math.ceil(math.sqrt(n))); ys = np.linspace(0, h - crop, g).astype(int); xs = np.linspace(0, w - crop, g).astype(int)
    return [(y, x) for y in ys for x in xs][:n]

# Build the two views every feature extractor works from:
#   * `crops`  — crop×crop windows at NATIVE resolution (pixel statistics untouched; images smaller than the crop are upscaled and flagged)
#   * `clip`   — the whole frame resized so its short side is clip_res, then centre-cropped (what CLIP sees)
# Nothing is ever squash-resized: that would alter exactly the noise / spectrum properties the forensic features measure.
def make_views(img, crop, n_crops, clip_res):
    """-> (list of native-res crops, clip view HxWx3, upscaled_flag). Images smaller than the crop are upscaled (flagged) rather than padded."""
    h, w = img.shape[:2]; up = False
    if min(h, w) < crop:
        s = crop / min(h, w); img = cv2.resize(img, (int(math.ceil(w * s)), int(math.ceil(h * s))), interpolation=cv2.INTER_CUBIC); h, w = img.shape[:2]; up = True
    crops = [img[y:y + crop, x:x + crop] for y, x in crop_positions(h, w, crop, n_crops)]
    s = clip_res / min(h, w); ch, cw = max(clip_res, int(round(h * s))), max(clip_res, int(round(w * s)))
    clip = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    y0, x0 = (ch - clip_res) // 2, (cw - clip_res) // 2
    return crops, clip[y0:y0 + clip_res, x0:x0 + clip_res], up

# --- physics trio helpers ---
# Grey-scale float image in [0,1].
def _gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

# Structure tensor at gradient scale sg with a Gaussian window sw: returns the gradients, the two eigenvalues (l1 >= l2),
# the dominant orientation and the 'coherence' (l1-l2)/(l1+l2) — 1 for a perfectly oriented edge, 0 for isotropic texture/noise.
def _structure_tensor(g, sg, sw):
    Ix = ndimage.gaussian_filter(g, sg, order=(0, 1)); Iy = ndimage.gaussian_filter(g, sg, order=(1, 0))
    Jxx = ndimage.gaussian_filter(Ix * Ix, sw); Jyy = ndimage.gaussian_filter(Iy * Iy, sw); Jxy = ndimage.gaussian_filter(Ix * Iy, sw)
    disc = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2)
    l1 = 0.5 * (Jxx + Jyy + disc); l2 = np.maximum(0.5 * (Jxx + Jyy - disc), 0)
    theta = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
    coh = (l1 - l2) / (l1 + l2 + 1e-8)
    return Ix, Iy, l1, l2, theta, coh

# Mean of a map inside each cell of a grid×grid patch layout (so a texture statistic is summarised over the image).
def _patch_means(m, grid):
    H, W = m.shape; ph, pw = max(1, H // grid), max(1, W // grid); g_h, g_w = H // ph, W // pw
    return m[:g_h * ph, :g_w * pw].reshape(g_h, ph, g_w, pw).mean(axis=(1, 3)).ravel()

# Apply fn (mean / std) inside every b×b block and return the per-block values as a flat array.
def _block_stat(m, b, fn):
    H, W = m.shape; g_h, g_w = H // b, W // b
    x = m[:g_h * b, :g_w * b].reshape(g_h, b, g_w, b)
    return fn(x, axis=(1, 3)).ravel()

# Rank correlation with a guard for constant inputs (returns 0 instead of NaN).
def _spearman(a, b):
    if len(a) < 4 or np.std(a) < 1e-12 or np.std(b) < 1e-12: return 0.0
    r = stats.spearmanr(a, b).correlation; return 0.0 if np.isnan(r) else float(r)

# === Physics trio (adapted multiLID) on one native-resolution crop ===
# Three physical regularities of camera images that convolutional / diffusion synthesis tends to violate.
def physics_features(rgb, grid=4, scales=(1.0, 2.0, 4.0), block=8):
    g = _gray(rgb); f = {}
    Ix, Iy, l1, l2, theta0, coh0 = _structure_tensor(g, scales[0], 2 * scales[0])
    # (1) ANISOTROPY. Real surfaces project to strongly oriented gradients; we look at the coherence of the structure tensor over a patch grid.
    # 'tex' = the textured half of the pixels (above-median gradient energy); flat regions would only add noise to the eigen-ratio.
    tex = l1 > np.median(l1)
    ratio = np.log10((l1 + 1e-8) / (l2 + 1e-8))
    P = _patch_means(coh0, grid)
    f['aniso_coh_mean'], f['aniso_coh_std'] = float(P.mean()), float(P.std())
    f['aniso_coh_p10'], f['aniso_coh_p90'] = float(np.percentile(P, 10)), float(np.percentile(P, 90))
    f['aniso_logratio_tex'] = float(ratio[tex].mean()); f['aniso_frac_gt10'] = float((ratio[tex] > 1).mean())
    f['aniso_coh_tex_minus_flat'] = float(coh0[tex].mean() - coh0[~tex].mean())
    # (2) CROSS-SCALE CONSISTENCY. For real structure the dominant orientation is the same at sigma 1, 2 and 4;
    # multi-resolution generators break this. We measure the coherence-weighted mean of cos(2*delta_theta) between neighbouring scales
    # (the factor 2 makes an orientation and its opposite count as equal).
    thetas, cohs = [theta0], [coh0]
    for s in scales[1:]:
        _, _, _, _, th, co = _structure_tensor(g, s, 2 * s); thetas.append(th); cohs.append(co)
    for i in range(len(scales) - 1):
        w = np.sqrt(cohs[i] * cohs[i + 1]); cons = np.cos(2 * (thetas[i] - thetas[i + 1]))
        f[f'xscale_cons_{i}'] = float((w * cons).sum() / (w.sum() + 1e-8))
        f[f'xscale_cons_{i}_pstd'] = float(_patch_means(cons, grid).std())
    # (3) NOISE-SIGNAL DECOUPLING. Sensor noise is independent of scene content and its variance grows with brightness (shot noise);
    # learned upsampling couples its residual to edges instead. Residual = image minus a smoothed version (Gaussian and median variants),
    # then per 8x8 block: residual energy vs gradient energy (rho_grad), vs mean intensity (rho_int), flat/textured ratio, level, and
    # the slope of noise variance against brightness inside flat blocks.
    gm = np.hypot(Ix, Iy)
    sig = _block_stat(gm, block, np.mean); inten = _block_stat(g, block, np.mean)
    for name, res in (('g', g - ndimage.gaussian_filter(g, 1.0)), ('m', g - ndimage.median_filter(g, 3))):
        ne = _block_stat(res, block, np.std)
        f[f'nsd_{name}_rho_grad'] = _spearman(ne, sig); f[f'nsd_{name}_rho_int'] = _spearman(ne, inten)
        lo, hi = sig <= np.percentile(sig, 20), sig >= np.percentile(sig, 80)
        f[f'nsd_{name}_flat_tex_ratio'] = float(ne[lo].mean() / (ne[hi].mean() + 1e-8))
        f[f'nsd_{name}_level'] = float(ne.mean())
        if lo.sum() >= 4 and np.std(inten[lo]) > 1e-6:   # shot-noise slope: noise variance vs brightness in flat blocks
            f[f'nsd_{name}_shot_slope'] = float(np.polyfit(inten[lo], ne[lo] ** 2, 1)[0])
        else: f[f'nsd_{name}_shot_slope'] = 0.0
    # Bonus: correlation of the residual noise between colour channels — a fingerprint of camera demosaicing that generators do not reproduce.
    ch = rgb.astype(np.float32) / 255.0
    resc = [ch[..., i] - ndimage.gaussian_filter(ch[..., i], 1.0) for i in range(3)]
    for (i, j, nm) in ((0, 1, 'rg'), (1, 2, 'gb'), (0, 2, 'rb')):
        a, b = resc[i].ravel(), resc[j].ravel()
        f[f'res_corr_{nm}'] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else 0.0
    return f

# === FFT features: the shape of the power spectrum ===
# Hann window to suppress edge leakage, 2-D FFT of the mean-removed image, log power spectrum.
def fft_features(rgb, n_bins=32):
    g = _gray(rgb); H, W = g.shape
    win = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2((g - g.mean()) * win)); P = np.abs(F) ** 2; logP = np.log1p(P)
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]; r = np.sqrt(((yy - cy) / max(cy, 1)) ** 2 + ((xx - cx) / max(cx, 1)) ** 2)
    # Azimuthal average: r is the distance from the spectrum centre, normalised so r = 1 is the Nyquist frequency along an axis.
    # The log power is averaged in n_bins radial rings (relative to the lowest ring so brightness/contrast cancel out).
    mask = r <= 1.0
    b = np.minimum((r * n_bins).astype(int), n_bins - 1)
    prof = np.bincount(b[mask], logP[mask], minlength=n_bins) / np.maximum(np.bincount(b[mask], minlength=n_bins), 1)
    prof = prof - prof[0]
    f = {f'fft_b{i:02d}': float(prof[i]) for i in range(n_bins)}
    # Summary scalars: the log-log spectral slope, the share of energy below 1/4 and above 3/4 Nyquist, and peak-to-baseline
    # contrasts at 1/2, 1/4, 1/8 Nyquist (where upsampling layers leave periodic grids).
    rmid = (np.arange(n_bins) + 0.5) / n_bins
    f['fft_slope'] = float(np.polyfit(np.log(rmid[2:]), prof[2:], 1)[0])
    tot = P[mask].sum() + 1e-12
    f['fft_low_share'] = float(P[mask & (r < 0.25)].sum() / tot); f['fft_high_share'] = float(P[mask & (r > 0.75)].sum() / tot)
    for frac in (0.5, 0.25, 0.125):
        k = min(max(int(frac * n_bins), 2), n_bins - 3)
        f[f'fft_peak_{frac}'] = float(prof[k] - 0.5 * (prof[k - 2] + prof[k + 2]))
    # Peaks exactly on the Nyquist axes / corner (after fftshift they sit at index 0): the checkerboard signature of transposed convolutions.
    nb = lambda y, x: logP[max(y - 3, 0):y + 4, max(x - 3, 0):x + 4].mean()
    f['fft_nyq_axis'] = float(0.5 * (logP[cy, 0] + logP[0, cx]) - 0.5 * (nb(cy, 0) + nb(0, cx)))
    f['fft_nyq_corner'] = float(logP[0, 0] - nb(0, 0))
    return f

# === ZED-lite: a tiny lossless image coder, the fallback when the pretrained SReC coder cannot be loaded ===
# It predicts every pixel of an image from the 2x-downsampled image (upsampled back), as a discretised logistic distribution.
# Trained by maximum likelihood on REAL photos only, so its 'surprise' on generated pixels is the feature.
class ZedLite(torch.nn.Module if torch else object):
    def __init__(self, ch=64, n_layers=5):
        super().__init__(); L = [torch.nn.Conv2d(3, ch, 3, padding=1), torch.nn.ReLU()]
        for _ in range(n_layers - 1): L += [torch.nn.Conv2d(ch, ch, 3, padding=1), torch.nn.ReLU()]
        L += [torch.nn.Conv2d(ch, 6, 3, padding=1)]; self.net = torch.nn.Sequential(*L)
    # ctx: the upsampled low-resolution image in [0,1]. Output: predicted mean (as a residual on ctx) and log-scale per sub-pixel.
    def forward(self, ctx):
        out = self.net(ctx * 2 - 1); return ctx + 0.1 * out[:, :3], torch.clamp(out[:, 3:] - 3.0, min=LOG_SCALE_MIN)

# log P(x) under a logistic distribution discretised to 256 grey levels (PixelCNN++ style, with the edge bins absorbing the tails
# and a density approximation where a bin's mass is too small to compute stably).
def disc_logistic_logprob(x, mean, log_scale, levels=256):
    F_ = torch.nn.functional; half = 0.5 / (levels - 1); inv_s = torch.exp(-log_scale); c = x - mean
    plus_in, min_in = inv_s * (c + half), inv_s * (c - half)
    cdf_delta = torch.sigmoid(plus_in) - torch.sigmoid(min_in)
    log_cdf_plus, log_one_minus_cdf_min = plus_in - F_.softplus(plus_in), -F_.softplus(min_in)
    mid_in = inv_s * c; log_pdf_mid = mid_in - log_scale - 2.0 * F_.softplus(mid_in)
    return torch.where(x < half, log_cdf_plus, torch.where(x > 1 - half, log_one_minus_cdf_min,
           torch.where(cdf_delta > 1e-5, torch.log(torch.clamp(cdf_delta, min=1e-12)), log_pdf_mid - math.log((levels - 1) / 2.0))))

@torch.no_grad()
# Exact entropy of that discretised distribution, evaluated by summing p*log p over all 256 levels (in chunks to bound memory).
# The ZED feature is NLL minus this entropy: 'how much cheaper than expected was this pixel to encode'.
def disc_logistic_entropy(mean, log_scale, levels=256, chunk=64):
    ent = torch.zeros_like(mean); grid = torch.linspace(0, 1, levels, device=mean.device)
    for i in range(0, levels, chunk):
        v = grid[i:i + chunk].view(1, 1, 1, 1, -1).expand(*mean.shape, -1)
        lp = disc_logistic_logprob(v, mean.unsqueeze(-1), log_scale.unsqueeze(-1), levels); ent -= (lp.exp() * lp).sum(-1)
    return ent

# Multi-resolution pairs (target, context): the image at each scale together with its 2x-downsampled-then-upsampled version.
def make_pyramid(x, n_scales=3):
    F_ = torch.nn.functional; pairs, cur = [], x
    for s in range(n_scales):
        low = F_.avg_pool2d(cur, 2); pairs.append((cur, F_.interpolate(low, scale_factor=2, mode='bilinear', align_corners=False))); cur = low
    return pairs


# =====================================================================================================================
#  Detector: rebuilds the full pipeline from an export folder written by the notebook's export cell
# =====================================================================================================================
# Registry so a transform can be applied by name (the name is what the training rows recorded in their 'aug' column).
AUG_FNS = {'jpeg': aug_jpeg, 'blur': aug_blur, 'resize': aug_resize, 'noise': aug_noise, 'jitter': aug_jitter, 'crop': aug_crop}
AUG_HELP = {'jpeg': 'JPEG re-encode at quality q', 'blur': 'Gaussian blur, sigma in px', 'resize': 'downscale by factor then upscale back (area / cubic)',
            'noise': 'additive Gaussian noise, sigma in [0,1] units', 'jitter': 'brightness/contrast/saturation each in ±a', 'crop': 'centre crop keeping fraction f of each side'}


# Apply one named transform; the seed makes the random ones (noise, jitter) reproducible.
def apply_aug(img, aug_type, aug_level, seed=0):
    """Same transform implementations as training. Returns (image, params)."""
    if aug_type == 'none':
        return img, {}
    rng = np.random.default_rng(int(seed))
    return AUG_FNS[aug_type](img, aug_level, rng)


# === Levelling: the first thing that happens to EVERY image, at training and at inference ===
# Rotate by EXIF, force RGB, cap the long side, and re-encode once as JPEG q95 (4:2:0). Real photos and generator PNGs then share
# the same compression history, so 'has JPEG artefacts' cannot act as the label. Done in memory here (the notebook wrote a file).
def level_image(src, q=95, max_side=1024):
    """Training-time levelling, in memory: EXIF transpose, RGB, long side <= max_side, one JPEG q re-encode (4:2:0). Returns HxWx3 uint8."""
    img = src if isinstance(src, Image.Image) else Image.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
    img = ImageOps.exif_transpose(img).convert('RGB')
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / max(w, h); img = img.resize((max(8, round(w * s)), max(8, round(h * s))), Image.LANCZOS)
    buf = io.BytesIO(); img.save(buf, 'JPEG', quality=q, subsampling=2)
    return np.array(Image.open(io.BytesIO(buf.getvalue())).convert('RGB'))


# Forest votes -> log-odds, clipped away from 0/1 so the logit stays finite. This is the single input of the Platt calibrator.
def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


# ======================================================================================================================
# Detector: loads the exported model folder and reproduces the training pipeline for a single image.
# Heavy public models (CLIP, VAEs, VGG, SReC) are loaded lazily on first use so cheap bundles never pay for them.
# ======================================================================================================================
class Detector:
    """Usage: d = Detector('aigc_export'); d.predict(image_or_path) -> dict"""

    def __init__(self, export_dir, device=None, zed_stride=None, quiet=False):
        self.dir = Path(export_dir); self.quiet = quiet
        # meta.json is the contract with the notebook: config knobs, the exact feature column order, which ZED backend and VAEs were
        # used, training medians for imputation, and the file names of every pickled object.
        self.meta = json.load(open(self.dir / 'meta.json'))
        self.C, self.tier, self.cols = self.meta['config'], self.meta['tier'], self.meta['feature_cols']
        self.smoke = bool(self.meta.get('smoke_test', False))
        self.tier['physics']['scales'] = tuple(self.tier['physics']['scales'])
        self.zed_stride = int(zed_stride if zed_stride is not None else self.C.get('ZED_ENT_STRIDE', 1))
        # The trained pieces: forest, the two calibrators, CLIP probe, CLIP PCA-16, cluster PCA-50 + k-means model + centroids,
        # and the small sub-forest used for SHAP explanations.
        f = self.meta['files']; L = lambda k: joblib.load(self.dir / f[k])
        self.rf, self.platt, self.iso, self.probe, self.pca16 = L('rf'), L('platt'), L('isotonic'), L('probe'), L('pca16')
        self.cluster_pca, self.cluster_model = L('cluster_pca'), L('cluster_model')
        self.cluster_centroids = np.load(self.dir / f['cluster_centroids']); self.cluster_method = self.meta['cluster_method']; self.K = int(self.meta['K'])
        self.shap_forest = joblib.load(self.dir / f['shap_forest']) if (self.dir / f['shap_forest']).exists() else None
        self.medians = self.meta.get('train_medians', {})
        self.device = device or ('cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu')
        self._clip = {}; self._aero = {}; self._zed = {}; self._shap = None
        self.zed_backend = self.meta.get('zed_backend'); self.vaes_used = self.meta.get('vaes_used', [])
        # Which extractors this bundle needs is decided from its feature list: a 'no-zed' bundle has no zed_* columns, so the SReC
        # coder is never loaded; a bundle without aero_* columns never loads the VAEs.
        want_zed = any(c.startswith('zed_') for c in self.cols); want_aero = any(c.startswith('aero_') for c in self.cols)
        self.use_zed, self.use_aero = want_zed, want_aero
        # joblib pickles are scikit-learn-version sensitive; warn loudly if the environment differs from the training one.
        v = self.meta.get('versions', {})
        try:
            import sklearn
            if v.get('sklearn') and v['sklearn'] != sklearn.__version__ and not quiet:
                print(f'[warn] bundle was trained with scikit-learn {v["sklearn"]}, this environment has {sklearn.__version__}; pickles may not load identically')
        except Exception:
            pass
        if not quiet:
            print(f'Detector: {len(self.cols)} features, K={self.K} clusters ({self.cluster_method}), ZED={self.zed_backend}, VAEs={self.vaes_used}, device={self.device}, smoke={self.smoke}')

    # ------------------------------------------------------------------ heavy models (lazy)
    # CLIP ViT-L/14 image embedding (768-d, L2-normalised). In SMOKE_TEST bundles a fixed random projection stands in for CLIP.
    def _clip_embed(self, imgs_u8):
        if self.smoke:
            if 'proj' not in self._clip: self._clip['proj'] = np.random.default_rng(0).normal(size=(3 * 16 * 16, 768)).astype(np.float32)
            x = np.stack([cv2.resize(i, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255 for i in imgs_u8])
            e = x @ self._clip['proj']; return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
        if 'model' not in self._clip:
            from transformers import CLIPModel, CLIPProcessor
            dt = torch.float16 if self.device == 'cuda' else torch.float32
            self._clip['model'] = CLIPModel.from_pretrained(self.C['CLIP_MODEL'], torch_dtype=dt).to(self.device).eval()
            self._clip['proc'] = CLIPProcessor.from_pretrained(self.C['CLIP_MODEL'])
        m = self._clip['model']
        with torch.no_grad():
            inp = self._clip['proc'](images=[Image.fromarray(i) for i in imgs_u8], return_tensors='pt')['pixel_values'].to(self.device, m.dtype)
            # Call the vision tower + projection explicitly rather than get_image_features(): newer transformers return an output object
            # from the latter, and this form is identical numerically across versions.
            pooled = m.vision_model(pixel_values=inp).pooler_output      # version-proof (newer transformers return an output object from get_image_features)
            e = torch.nn.functional.normalize(m.visual_projection(pooled).float(), dim=-1)
        return e.cpu().numpy()

    # AEROBLADE: load the latent-diffusion VAEs the bundle was trained with (only those whose features exist) plus LPIPS (VGG16).
    # A VAE that fails to load (e.g. gated FLUX repo without a token) is skipped and its features are later imputed with training medians.
    def _load_aero(self):
        if self._aero: return
        if self.smoke: self._aero['vaes'] = {'mock': None}; return
        from diffusers import AutoencoderKL
        import lpips
        tok = self.C.get('HF_TOKEN'); self._aero['vaes'] = {}
        for name, spec in self.C['VAES'].items():
            if name not in self.vaes_used: continue
            repo, sub = spec[0], spec[1]
            try:
                kw = dict(torch_dtype=torch.float32); kw.update(dict(subfolder=sub) if sub else {}); kw.update(dict(token=tok) if tok else {})
                vae = AutoencoderKL.from_pretrained(repo, **kw).to(self.device).eval()
                # Decode one image at a time inside the VAE: ~4x lower peak memory, identical numbers.
                if hasattr(vae, 'enable_slicing'): vae.enable_slicing()
                self._aero['vaes'][name] = vae
            except Exception as e:
                print(f'[warn] VAE {name} failed to load ({repo}): {e} -> its features will be imputed with training medians')
        self._aero['lp'] = lpips.LPIPS(net='vgg', verbose=False).to(self.device).eval()

    @torch.no_grad() if torch is not None else (lambda f: f)
    # Encode -> decode each crop through every VAE and measure how much it changed: LPIPS (total and VGG layer 2, as in the AEROBLADE
    # paper), pixel MSE and cosine between VGG layer-2 features. Generated images survive the round trip almost losslessly; photos do not.
    # The min over VAEs of the layer-2 LPIPS is added as an extra feature ('which decoder reproduces this best').
    def _aero_feats(self, crops):
        self._load_aero(); feats = [{} for _ in crops]
        x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(self.device)
        if self.smoke:
            rec = torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1); d = ((x - rec) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
            for i in range(len(feats)):
                feats[i].update({'aero_mock_lpips': float(d[i]), 'aero_mock_lpips_l2': float(d[i] * 0.5), 'aero_mock_mse': float(d[i]), 'aero_mock_cos': float(1 - d[i]), 'aero_min_lpips_l2': float(d[i] * 0.5)})
            return feats
        lp = self._aero['lp']; l2_all = []
        for name, vae in self._aero['vaes'].items():
            z = vae.encode(x).latent_dist.mode(); rec = vae.decode(z).sample.clamp(-1, 1)
            tot, per_layer = lp(x, rec, retPerLayer=True)
            fx = lp.net.forward(lp.scaling_layer(x))[1]; fr = lp.net.forward(lp.scaling_layer(rec))[1]
            cos = torch.nn.functional.cosine_similarity(fx.flatten(1), fr.flatten(1), dim=1); mse = ((x - rec) ** 2).mean(dim=(1, 2, 3))
            l2 = per_layer[1].flatten(); l2_all.append(l2)
            for i in range(len(feats)):
                feats[i].update({f'aero_{name}_lpips': float(tot.flatten()[i]), f'aero_{name}_lpips_l2': float(l2[i]), f'aero_{name}_mse': float(mse[i]), f'aero_{name}_cos': float(cos[i])})
        if l2_all:
            mins = torch.stack(l2_all).min(dim=0).values
            for i in range(len(feats)): feats[i]['aero_min_lpips_l2'] = float(mins[i])
        return feats

    # ZED readout: returns a function crop -> per-scale {NLL, entropy, gap} statistics.
    # Backend 'srec': the pretrained SReC coder (its code is cloned from GitHub once; weights ship in the bundle).
    # Backend 'lite': the small ZED-lite coder trained in the notebook on reserved real images.
    def _zed_readout(self):
        if 'readout' in self._zed: return self._zed['readout']
        st = self.zed_stride
        if self.zed_backend == 'srec':
            srec_dir = self.dir / 'SReC'
            # Look for the SReC code inside the bundle first, then next to the scripts (one clone can serve every bundle), then clone it.
            shared = Path(__file__).resolve().parent / 'SReC'                # one clone next to the scripts serves every bundle
            if not srec_dir.exists() and (shared / 'src').exists(): srec_dir = shared
            if not srec_dir.exists():
                try:
                    subprocess.run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/caoscott/SReC.git', str(srec_dir)], check=True)
                except FileNotFoundError:
                    raise RuntimeError(f'git is not installed (needed once to fetch the SReC coder code). Install Git for Windows / git, or copy an existing SReC folder to {srec_dir}') from None
            sys.path.insert(0, str(srec_dir))
            from src import configs, network
            from src.l3c import logistic_mixture as lm
            configs.n_feats, configs.resblocks, configs.K, configs.scale = 64, 3, 10, 3
            configs.log_likelihood, configs.collect_probs = True, True
            comp = network.Compressor()
            wpath = self.dir / self.meta['files']['srec_weights']
            if not wpath.exists(): wpath = srec_dir / 'models' / 'openimages.pth'
            comp.nets.load_state_dict(torch.load(str(wpath), map_location='cpu', weights_only=False)['nets']); comp = comp.to(self.device).eval()
            loss_fn = lm.DiscretizedMixLogisticLoss(rgb_scale=True)

            @torch.no_grad()
            # Entropy of SReC's 10-component logistic MIXTURE per sub-pixel: evaluate the mixture at all 256 grey levels (chunked) and sum p*log p.
            # The conditioning on the true value of earlier colour channels mirrors what the real coder does.
            def dml_entropy(x_true, l, chunk=32):
                x, logit_pis, means, log_scales, K = loss_fn._extract_non_shared(x_true, l)
                log_w = torch.log_softmax(logit_pis, dim=2); N, Cc, K_, H, W = means.shape
                ent = torch.zeros(N, Cc, H, W, device=l.device); levels = torch.arange(0, 256, dtype=torch.float32, device=l.device)
                for i in range(0, 256, chunk):
                    v = levels[i:i + chunk].view(1, 1, 1, 1, 1, -1).expand(N, Cc, K_, H, W, -1)
                    lp = loss_fn.log_cdf(v, v, means.unsqueeze(-1), log_scales.unsqueeze(-1))
                    lp = torch.logsumexp(log_w.unsqueeze(-1) + lp, dim=2); ent -= (lp.exp() * lp).sum(-1)
                return ent

            @torch.no_grad()
            # SReC path: run the compressor once; for every coded slice at every scale compute NLL and entropy, optionally on a pixel stride
            # (the speed knob), and summarise the gap (NLL - entropy) per scale: mean, |mean|, std.
            def readout(x_u8):
                x = torch.from_numpy(x_u8).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                bits = comp(x); per = {}
                for (y_i, lm_probs, levels) in bits.probs:
                    if lm_probs is None: continue
                    y_s, l_s = y_i[..., ::st, ::st].contiguous(), lm_probs.probs[..., ::st, ::st].contiguous()
                    nll = loss_fn(y_s, l_s); ent = dml_entropy(y_s, l_s)
                    s = int(lm_probs.name.split('/')[1].split('_')[0]); per.setdefault(s, []).append((nll.flatten(), ent.flatten()))
                f = {}
                for s, pairs in per.items():
                    nll = torch.cat([p[0] for p in pairs]); ent = torch.cat([p[1] for p in pairs]); gap = nll - ent
                    f.update({f'zed_s{s}_nll': nll.mean().item(), f'zed_s{s}_ent': ent.mean().item(), f'zed_s{s}_gap': gap.mean().item(),
                              f'zed_s{s}_gap_abs': gap.abs().mean().item(), f'zed_s{s}_gap_std': gap.std().item()})
                return f
        else:
            model = ZedLite().to(self.device)
            model.load_state_dict(torch.load(str(self.dir / self.meta['files']['zedlite']), map_location=self.device)); model.eval()

            @torch.no_grad()
            # ZED-lite path: same statistics from the small coder's own pyramid.
            def readout(x_u8):
                x = torch.from_numpy(x_u8).permute(2, 0, 1).unsqueeze(0).float().div(255).to(self.device); f = {}
                for s, (target, ctx) in enumerate(make_pyramid(x)):
                    mean, ls = model(ctx); mean, ls, target = mean[..., ::st, ::st], ls[..., ::st, ::st], target[..., ::st, ::st]
                    nll = -disc_logistic_logprob(target, mean, ls); ent = disc_logistic_entropy(mean, ls); gap = nll - ent
                    f.update({f'zed_s{s}_nll': nll.mean().item(), f'zed_s{s}_ent': ent.mean().item(), f'zed_s{s}_gap': gap.mean().item(),
                              f'zed_s{s}_gap_abs': gap.abs().mean().item(), f'zed_s{s}_gap_std': gap.std().item()})
                return f
        self._zed['readout'] = readout; return readout

    # ------------------------------------------------------------------ features
    # === The full feature row for one levelled image, in the exact column order the forest was trained on ===
    # 1. views  2. physics + FFT on each crop (averaged over crops)  3. AEROBLADE  4. ZED  5. CLIP: PCA-16 + probe logit
    # 6. cluster id (nearest centroid in the cluster PCA space) as one-hot  7. reindex to the training columns, impute NaN / missing with medians.
    def featurize(self, img_levelled):
        """Levelled RGB uint8 -> (feature row DataFrame with the training columns, cluster id, extras dict)."""
        tc = self.tier; crops, clip_v, up = make_views(img_levelled, tc['crop'], tc['n_crops'], tc['clip_res'])
        f = {}
        fs = [dict(**physics_features(c, **tc['physics']), **fft_features(c, tc['fft_bins'])) for c in crops]
        f.update({k: float(np.mean([x[k] for x in fs])) for k in fs[0]})
        if self.use_aero:
            fa = self._aero_feats(crops); f.update({k: float(np.mean([x[k] for x in fa])) for k in fa[0]})
        if self.use_zed:
            ro = self._zed_readout(); fz = [ro(c) for c in crops]; f.update({k: float(np.mean([x[k] for x in fz])) for k in fz[0]})
        e = self._clip_embed([clip_v])
        pcs = self.pca16.transform(e)[0]; f.update({f'clip_pc{j:02d}': float(v) for j, v in enumerate(pcs)})
        f['clip_probe_logit'] = float(self.probe.decision_function(e)[0])
        z = self.cluster_pca.transform(e)
        k = int(self.cluster_model.predict(z)[0]) if self.cluster_method in ('kmeans', 'gmm') else int(np.argmin(((z[:, None] - self.cluster_centroids[None]) ** 2).sum(-1)))
        # The unsupervised cluster enters the forest one-hot: this is how the forest can weight features differently per image sub-type.
        for j in range(self.K): f[f'cl_{j}'] = int(j == k)
        row = pd.DataFrame([f]).reindex(columns=self.cols)
        missing = [c for c in self.cols if c not in f]
        for c in missing: row[c] = self.medians.get(c, 0.0)
        row = row.replace([np.inf, -np.inf], np.nan).fillna(pd.Series(self.medians)).fillna(0.0)
        return row, k, dict(upscaled_for_crop=up, missing_features=missing)

    # Score an image: level (unless the caller already did), featurise, forest vote, Platt calibration, verdict at 0.5.
    def predict(self, src, explain=False, already_levelled=False):
        """src: path | bytes | PIL.Image | HxWx3 uint8 array. Returns dict with verdict and calibrated probability of being AI-generated."""
        img = src if (isinstance(src, np.ndarray) and already_levelled) else level_image(Image.fromarray(src) if isinstance(src, np.ndarray) else src,
                                                                                         q=self.C['LEVEL_JPEG_Q'], max_side=self.C['MAX_SIDE'])
        row, k, extra = self.featurize(img)
        p_raw = float(self.rf.predict_proba(row.values)[:, 1][0])
        p_cal = float(self.platt.predict_proba(_logit(np.array([p_raw]))[:, None])[0, 1])
        out = dict(verdict='ai' if p_cal > 0.5 else 'real', p_ai=round(p_cal, 4), p_ai_raw=round(p_raw, 4), cluster=k, **extra)
        if explain and self.shap_forest is not None:
            out['top_contributions'] = self.explain(row)
        return out

    # SHAP contributions from the sub-forest (first N trees, an unbiased but cheaper stand-in for the full forest).
    def explain(self, row, top=10):
        import shap
        if self._shap is None: self._shap = shap.TreeExplainer(self.shap_forest)
        v = self._shap.shap_values(row.values); v = v[1] if isinstance(v, list) else (v[:, :, 1] if v.ndim == 3 else v)
        s = pd.Series(v[0], index=self.cols).sort_values(key=np.abs, ascending=False)
        return {k: round(float(x), 4) for k, x in s.head(top).items()}

    def transform_levels(self):
        return {k: list(v) for k, v in self.C['AUG_LEVELS'].items()}
