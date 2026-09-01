#!/usr/bin/env python
"""
classify_folder.py — score every image in a folder with the exported detector and write a JSON report.

    python classify_folder.py /path/to/images --bundle ./aigc_export --out results.json
    python classify_folder.py /path/to/images --bundle ./aigc_export --recursive --explain
    python classify_folder.py /path/to/images --bundle ./aigc_export --fast        # ZED entropy on a 4x pixel stride (approximate, much faster on CPU)

Output JSON:
{
  "bundle": "...", "created": "...", "n_images": 12, "summary": {"real": 7, "ai": 5, "errors": 0},
  "results": [ {"file": "a.jpg", "verdict": "ai", "p_ai": 0.93, "p_ai_raw": 0.81, "cluster": 3, ...}, ... ]
}
p_ai is the Platt-calibrated probability that the image is AI-generated; verdict = "ai" if p_ai > 0.5 else "real".
"""
import argparse, json, sys, time
from pathlib import Path

# File types we treat as images when scanning the folder.
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def main():
    # Command-line options. --bundle points at the folder holding meta.json + the model files; --fast trades a little ZED
    # fidelity for a large CPU speed-up; --explain adds per-image SHAP contributions; --threshold moves the verdict cut-off.
    ap = argparse.ArgumentParser(description='Classify a folder of images as real or AI-generated.')
    ap.add_argument('folder', help='folder containing images')
    ap.add_argument('--bundle', default='aigc_export', help='export folder written by the notebook (default: ./aigc_export)')
    ap.add_argument('--out', default=None, help='output JSON path (default: <folder>/aigc_results.json)')
    ap.add_argument('--recursive', action='store_true', help='also scan sub-folders')
    ap.add_argument('--explain', action='store_true', help='add the top-10 SHAP feature contributions per image (slower)')
    ap.add_argument('--fast', action='store_true', help='ZED entropy on a 4x pixel stride (approximate; big speed-up on CPU)')
    ap.add_argument('--device', default=None, help="'cuda' or 'cpu' (default: auto)")
    ap.add_argument('--threshold', type=float, default=0.5, help='verdict threshold on the calibrated probability (default 0.5)')
    args = ap.parse_args()

    # Import the detector library from the same folder as this script, wherever the script is run from.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from aigc_inference import Detector

    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f'not a folder: {folder}')
    # Collect the images (optionally recursing into sub-folders); sorted so the output order is stable.
    files = sorted(p for p in (folder.rglob('*') if args.recursive else folder.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not files:
        sys.exit(f'no images found in {folder}')
    # Load the model once. Heavy extractors (CLIP, VAEs, SReC) are loaded lazily by the first prediction.
    det = Detector(args.bundle, device=args.device, zed_stride=(4 if args.fast else None))

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = lambda x, **k: x
    results, t0 = [], time.time()
    # Score every image; a failure on one file is recorded in the JSON instead of aborting the whole run.
    for p in tqdm(files, desc='classifying'):
        try:
            r = det.predict(str(p), explain=args.explain)
            r['verdict'] = 'ai' if r['p_ai'] > args.threshold else 'real'
            results.append({'file': str(p.relative_to(folder)), **r})
        except Exception as e:
            results.append({'file': str(p.relative_to(folder)), 'error': repr(e)})
    # Summary counts + the report itself (bundle, folder, threshold, timing, one entry per image).
    n_ai = sum(r.get('verdict') == 'ai' for r in results); n_real = sum(r.get('verdict') == 'real' for r in results); n_err = sum('error' in r for r in results)
    report = dict(bundle=str(Path(args.bundle).resolve()), folder=str(folder.resolve()), created=time.strftime('%Y-%m-%d %H:%M:%S'),
                  threshold=args.threshold, n_images=len(files), seconds=round(time.time() - t0, 1),
                  summary=dict(real=n_real, ai=n_ai, errors=n_err), results=results)
    out = Path(args.out) if args.out else folder / 'aigc_results.json'
    out.write_text(json.dumps(report, indent=2))
    print(f'\n{len(files)} images: {n_real} real, {n_ai} ai, {n_err} errors  ->  {out}')
    # Short human-readable preview of the first ten results; everything is in the JSON.
    for r in results[:10]:
        print(f"  {r['file']:40s} {r.get('verdict', 'ERROR'):5s} p_ai={r.get('p_ai', float('nan')):.3f}" if 'error' not in r else f"  {r['file']:40s} ERROR {r['error'][:60]}")
    if len(results) > 10: print(f'  … {len(results) - 10} more in the JSON')


if __name__ == '__main__':
    main()
