#!/usr/bin/env python
"""
make_eval_subset.py — build a labelled folder of evaluation images without downloading whole archives.

    python make_eval_subset.py --out eval_subset --n-real 300 --n-ai 300
    python make_eval_subset.py --out eval_subset --n-real 300 --n-ai 300 --real-source cocodataset   # reals from the official COCO val2017.zip

Sources (members are read out of the remote zips with HTTP Range requests, ~1 MB per image):
  real : WildFake  Images/Real/coco.zip                -> members whose path contains 'val2017'   (the organisers' COCO val2017 subset)
         or the official http://images.cocodataset.org/zips/val2017.zip  (--real-source cocodataset)
  ai   : WildFake  Images/Diffusion_based/DALLE.zip    -> members whose path contains 'Advanced'  (DALL·E 3, the organisers' 'DALL·E Advanced' subset)

Both are held out of the notebook's training data, so scores on this folder are honest.

Writes:
  <out>/<image files>            original bytes (the detector levels them itself, exactly as in training)
  <out>/labels.json              {"files": [...], "labels": [0/1 ...], "sources": [...], "members": [...]}   (0 = real, 1 = ai; same order)
  <out>/labels.csv               file,label,source
  <out>/labels.npy               the 0/1 array in the same order as labels.json["files"]
Feed <out> to classify_folder.py and <out>/labels.json to classify_folder_error_analysis.py.

Requires: pip install modelscope requests numpy   (modelscope only for the WildFake sources)
"""
import argparse, io, json, sys, time, zipfile, threading, hashlib, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
WILDFAKE_REPO = 'hy2628982280/WildFake'
COCO_OFFICIAL = 'http://images.cocodataset.org/zips/val2017.zip'


# === Read a zip archive over HTTP without downloading it ===
# A zip's table of contents lives at its END, so with HTTP Range requests we can fetch the directory (a few MB), pick members,
# and then fetch only those members' bytes. Blocks are cached and shared between threads.
class RemoteZip:
    """Random access to a zip over HTTP Range requests: only the table of contents and the sampled members are transferred."""
    BLOCK = 256 * 1024

    # A 1-byte Range probe tells us whether the server honours Range requests and how big the file is.
    def __init__(self, url, headers=None, cookies=None, name=None):
        self.url, self.headers, self.cookies, self.name = url, dict(headers or {}), cookies, name or url
        r = requests.get(url, headers={**self.headers, 'Range': 'bytes=0-0'}, cookies=cookies, allow_redirects=True, timeout=60)
        if r.status_code != 206:
            raise RuntimeError(f'HTTP {r.status_code}: server does not honour Range requests for {self.name}')
        self.size = int(r.headers['Content-Range'].split('/')[-1]); self.cache, self.lock = {}, threading.Lock()

    # One Range request for [start, end), with retries; the response must be 206 and exactly the requested length.
    def fetch(self, start, end):
        err = None
        for attempt in range(6):
            try:
                r = requests.get(self.url, headers={**self.headers, 'Range': f'bytes={start}-{end - 1}'}, cookies=self.cookies, timeout=180)
                if r.status_code == 206 and len(r.content) == end - start: return r.content
                err = f'HTTP {r.status_code}, {len(r.content)} bytes'
            except Exception as e:
                err = repr(e)
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f'range fetch failed for {self.name} [{start},{end}): {err}')

    # Cached 256 KB blocks for the small reads zipfile does while parsing headers.
    def block(self, i):
        with self.lock:
            if i in self.cache: return self.cache[i]
        data = self.fetch(i * self.BLOCK, min(self.size, (i + 1) * self.BLOCK))
        with self.lock:
            if len(self.cache) > 2048: self.cache.clear()
            self.cache[i] = data
        return data

    def open(self): return zipfile.ZipFile(_RangeFile(self))


# A seekable file-like view over RemoteZip, which is what zipfile.ZipFile expects. Big reads (member bodies) go straight to one
# exact Range request; small reads (headers, directory) go through the block cache.
class _RangeFile(io.RawIOBase):
    def __init__(self, rz): self.rz, self.pos = rz, 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.rz.size + off}[whence]; return self.pos
    def read(self, n=-1):
        end = self.rz.size if (n is None or n < 0) else min(self.rz.size, self.pos + n)
        if end <= self.pos: return b''
        B = self.rz.BLOCK
        if end - self.pos >= 2 * B: data = self.rz.fetch(self.pos, end)
        else:
            parts = []
            for i in range(self.pos // B, (end - 1) // B + 1):
                d = self.rz.block(i); parts.append(d[max(self.pos, i * B) - i * B: min(end, (i + 1) * B) - i * B])
            data = b''.join(parts)
        self.pos = end; return data
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


# Build the ModelScope download URL for one WildFake archive (with the API's auth headers/cookies) and wrap it as a RemoteZip.
def wildfake_zip(path):
    from modelscope_hub import HubApi
    api = HubApi().legacy
    return RemoteZip(api.get_download_url(WILDFAKE_REPO, 'dataset', path), headers=api._headers(), cookies=api._session.cookies, name=path)


# List the archive's image members, keep those whose path matches the regex (e.g. 'val2017' or 'advanced'), sample n with a seed.
def sample_members(rz, pattern, n, seed):
    zf = rz.open()
    names = [m for m in zf.namelist() if Path(m).suffix.lower() in IMG_EXTS and re.search(pattern, m, re.I)]
    zf.close()
    if not names: raise RuntimeError(f'no members matching /{pattern}/ in {rz.name}')
    rng = np.random.default_rng(seed)
    return [names[i] for i in rng.permutation(len(names))[:n]]


# Download the sampled members with a thread pool; each thread keeps its own ZipFile handle (they are not thread-safe).
# Files get neutral hashed names so nothing in the file name reveals the label.
def fetch_all(rz, members, out_dir, prefix, threads):
    tl = threading.local()
    def one(m):
        if not hasattr(tl, 'zf'): tl.zf = rz.open()
        data = tl.zf.open(m).read()
        name = f'{prefix}_{hashlib.md5(m.encode()).hexdigest()[:10]}{Path(m).suffix.lower()}'
        (out_dir / name).write_bytes(data); return name, m
    with ThreadPoolExecutor(threads) as ex:
        return list(ex.map(one, members))


def main():
    ap = argparse.ArgumentParser(description='Build a labelled evaluation folder from the held-out demo sources.')
    ap.add_argument('--out', default='eval_subset'); ap.add_argument('--n-real', type=int, default=300); ap.add_argument('--n-ai', type=int, default=300)
    ap.add_argument('--real-source', choices=['wildfake', 'cocodataset'], default='wildfake', help="wildfake = coco.zip/val2017 members (organisers' subset); cocodataset = official COCO val2017.zip")
    ap.add_argument('--ai-pattern', default=r'advanced', help="regex on member paths inside DALLE.zip (default: the 'Advanced' = DALL·E 3 folder)")
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Reals: WildFake's coco.zip members under val2017 (the organisers' subset) or, optionally, the official COCO val2017.zip.
    if a.real_source == 'wildfake':
        rz_real = wildfake_zip('Images/Real/coco.zip'); real_pat = r'val2017'; real_src = 'wildfake:coco/val2017'
    else:
        rz_real = RemoteZip(COCO_OFFICIAL, name='val2017.zip'); real_pat = r'.'; real_src = 'cocodataset:val2017'
    print(f'reading table of contents: {rz_real.name} ({rz_real.size/1e9:.2f} GB)')
    real_members = sample_members(rz_real, real_pat, a.n_real, a.seed)
    # AI images: DALLE.zip members under 'Advanced' (= DALL·E 3), the organisers' fake demo subset.
    rz_ai = wildfake_zip('Images/Diffusion_based/DALLE.zip')
    print(f'reading table of contents: {rz_ai.name} ({rz_ai.size/1e9:.2f} GB)')
    ai_members = sample_members(rz_ai, a.ai_pattern, a.n_ai, a.seed + 1)
    print(f'fetching {len(real_members)} real + {len(ai_members)} ai images with {a.threads} threads …')
    got_real = fetch_all(rz_real, real_members, out, 'img', a.threads)
    got_ai = fetch_all(rz_ai, ai_members, out, 'img', a.threads)

    files = [f for f, _ in got_real] + [f for f, _ in got_ai]
    labels = [0] * len(got_real) + [1] * len(got_ai)
    sources = [real_src] * len(got_real) + ['wildfake:DALLE/' + ('Advanced' if a.ai_pattern == 'advanced' else a.ai_pattern)] * len(got_ai)
    members = [m for _, m in got_real] + [m for _, m in got_ai]
    # Shuffle so the folder order carries no label information, then write labels.json / labels.csv / labels.npy in that order.
    order = np.random.default_rng(a.seed).permutation(len(files))            # shuffle so the folder order carries no label information
    files, labels, sources, members = [files[i] for i in order], [labels[i] for i in order], [sources[i] for i in order], [members[i] for i in order]
    json.dump({'files': files, 'labels': labels, 'sources': sources, 'members': members, 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
               'label_meaning': {'0': 'real', '1': 'ai'}}, open(out / 'labels.json', 'w'), indent=1)
    (out / 'labels.csv').write_text('file,label,source\n' + '\n'.join(f'{f},{l},{s}' for f, l, s in zip(files, labels, sources)))
    np.save(out / 'labels.npy', np.array(labels, dtype=np.int8))
    print(f'done: {len(files)} images in {out} ({sum(labels)} ai, {len(labels) - sum(labels)} real) in {(time.time() - t0) / 60:.1f} min')
    print(f'labels -> {out / "labels.json"}  (also labels.csv / labels.npy, same order as labels.json["files"])')


if __name__ == '__main__':
    main()
