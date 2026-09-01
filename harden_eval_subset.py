#!/usr/bin/env python
"""
harden_eval_subset.py — remove the two dataset shortcuts from an evaluation folder, so a re-score tells you whether the detector
is reading GENERATOR artefacts (what we want) or DATASET artefacts (what the organisers warned about).

    python harden_eval_subset.py eval_subset --out eval_subset_hardened
    python harden_eval_subset.py eval_subset --out eval_subset_hardened --quality 90 --long-side 640 --mode both

The problem this addresses
--------------------------
In the COCO-val2017-vs-DALL·E-3 evaluation the two classes differ in ways that have nothing to do with how the pixels were made:

  * compression history — COCO photos are JPEG files (compressed once by COCO) and the detector's levelling step re-encodes them
    as JPEG q95, so they are compressed TWICE; DALL·E images are PNGs, so after levelling they are compressed ONCE.
    Double-compression leaves detectable traces (blocking, quantisation-table interactions) that a classifier can use as
    "this is a real photo", even though it is a property of the dataset, not of photographs.
  * size / shape — COCO images are ~640×480 (small, non-square); DALL·E 3 images are 1024×1024 (large, square). The native
    256-px crop therefore covers a quarter of a DALL·E frame but most of a COCO frame, and the centre-crop for CLIP removes
    the sides of COCO images only. Either cue separates the classes perfectly without looking at a single artefact.

What this script does
---------------------
It writes a copy of the folder in which those cues are EQUALISED, then you score the copy with the same model:

  mode 'jpeg'   : every image (both classes) is decoded and saved as a JPEG at --quality. Now both classes have been
                  JPEG-compressed the same number of times before levelling (levelling adds one more to both).
  mode 'size'   : every image whose long side exceeds --long-side is downscaled (high-quality Lanczos) so that its long side
                  equals --long-side. DALL·E images become COCO-sized; COCO images (already ≤ 640) are untouched.
  mode 'both'   : size first, then jpeg (the default — removes both cues).

labels.json / labels.csv / labels.npy are copied unchanged (file names are preserved), so the same command as before evaluates it:

    python classify_folder_error_analysis.py eval_subset_hardened --labels eval_subset_hardened/labels.json --bundle . --out-dir eval_report_hardened

How to read the result
----------------------
  * accuracy stays close to the original           -> the model is reading generator artefacts that survive re-compression and
                                                      downscaling: a genuine detector on this pair.
  * accuracy drops a lot on the 'jpeg' copy         -> the model was largely using compression history ("double JPEG = real").
  * accuracy drops a lot on the 'size' copy         -> the model was largely using resolution / framing.
Run all three modes to attribute the effect. Caveat: downscaling by ~0.6 also genuinely destroys some high-frequency generator
artefacts, so a moderate drop on 'size' is partly legitimate degradation, not only shortcut removal — the 'jpeg' mode is the
cleaner test, because a q90 re-encode leaves generator artefacts largely intact (the training data already contained JPEG-q90
augmentations of fakes, so the model has seen this condition).
"""
import argparse, io, json, shutil
from pathlib import Path
from PIL import Image, ImageOps

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


# Apply the chosen equalisation to one image: optional downscale to a common long side (removes the size/shape cue),
# optional JPEG re-encode (removes the compression-history cue). Returns the bytes plus the extension to save under.
def harden(img, mode, quality, long_side):
    """PIL image -> (bytes, extension) after the chosen equalisation."""
    img = ImageOps.exif_transpose(img).convert('RGB')
    if mode in ('size', 'both'):
        w, h = img.size
        if max(w, h) > long_side:
            s = long_side / max(w, h); img = img.resize((max(8, round(w * s)), max(8, round(h * s))), Image.LANCZOS)
    if mode in ('jpeg', 'both'):
        buf = io.BytesIO(); img.save(buf, 'JPEG', quality=quality, subsampling=2); return buf.getvalue(), '.jpg'
    buf = io.BytesIO(); img.save(buf, 'PNG'); return buf.getvalue(), '.png'


def main():
    ap = argparse.ArgumentParser(description='Equalise compression history and/or size across an evaluation folder.')
    ap.add_argument('folder'); ap.add_argument('--out', required=True)
    ap.add_argument('--mode', choices=['jpeg', 'size', 'both'], default='both')
    ap.add_argument('--quality', type=int, default=90, help='JPEG quality for the re-encode (90 = a level the model was trained with as augmentation)')
    ap.add_argument('--long-side', type=int, default=640, help='target long side for the size equalisation (COCO images are ~640 px)')
    a = ap.parse_args()
    src, dst = Path(a.folder), Path(a.out); dst.mkdir(parents=True, exist_ok=True)
    # Walk the images in labels.json order so the rewritten labels stay aligned; count what was actually changed.
    labels = json.load(open(src / 'labels.json'))
    new_files, stats = [], {'reencoded': 0, 'downscaled': 0}
    for f in labels['files']:
        img = Image.open(src / f); w, h = img.size
        data, ext = harden(img, a.mode, a.quality, a.long_side)
        stats['reencoded'] += ext == '.jpg'; stats['downscaled'] += (a.mode in ('size', 'both')) and max(w, h) > a.long_side
        # keep the same stem so labels stay aligned; the extension may change (png -> jpg) so labels.json is rewritten with the new names
        # Keep the file stem; the extension may become .jpg, so labels.json is rewritten with the new names.
        name = Path(f).stem + ext; (dst / name).write_bytes(data); new_files.append(name)
    # Record what was done inside labels.json so a report can say which version of the folder it scored.
    labels['files'] = new_files; labels['hardened'] = dict(mode=a.mode, quality=a.quality, long_side=a.long_side, source_folder=str(src))
    json.dump(labels, open(dst / 'labels.json', 'w'), indent=1)
    (dst / 'labels.csv').write_text('file,label,source\n' + '\n'.join(f'{f},{l},{s}' for f, l, s in zip(new_files, labels['labels'], labels.get('sources', ['?'] * len(new_files)))))
    if (src / 'labels.npy').exists(): shutil.copy(src / 'labels.npy', dst / 'labels.npy')
    print(f"{len(new_files)} images -> {dst}  (mode={a.mode}: {stats['reencoded']} re-encoded as JPEG q{a.quality}, {stats['downscaled']} downscaled to long side {a.long_side})")
    print(f'score it with:  python classify_folder_error_analysis.py {dst} --labels {dst / "labels.json"} --bundle . --out-dir {dst.name}_report')


if __name__ == '__main__':
    main()
