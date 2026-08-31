#!/usr/bin/env python
"""
test_leakage.py — prove that the local evaluation images were never part of training.

    python test_leakage.py --cache aigc_rf_cache-20260829T163203Z-1-001.zip --labels eval_subset/labels.json

The WildFake plan (features/plan_wildfake.parquet inside the training cache) lists every WildFake member the notebook fetched,
with a demo_group column: rows with a demo_group (coco_val2017 / dalle_advanced) were held out of EVERY fitted component and only
used for the demo-set report; rows without one (and not flagged zed_holdout) are the training pool.  make_eval_subset.py records
the member path of every evaluation image in labels.json["members"].  Intersecting the two answers the question directly.
"""
import argparse, io, json, zipfile
from pathlib import Path
import pandas as pd


def read_member(cache, member):
    """Read one file from the cache, whether it is the Drive zip or an unzipped folder (nothing is extracted)."""
    cache = Path(cache)
    if cache.is_dir():
        return (cache / member).read_bytes()
    with zipfile.ZipFile(cache) as z:
        hits = [n for n in z.namelist() if n.replace('\\', '/').endswith('/' + member) or n == member]
        if not hits: raise SystemExit(f'{member} not found in {cache}')
        return z.read(hits[0])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cache', required=True); ap.add_argument('--labels', required=True); a = ap.parse_args()

    # The plan: every WildFake member the notebook decided to fetch, with its role.
    plan = pd.read_parquet(io.BytesIO(read_member(a.cache, 'features/plan_wildfake.parquet')))
    # Training pool = members with no demo group that were not reserved for ZED-lite (those never enter the forest's data either).
    train_members = set(plan.loc[plan.demo_group.isna() & ~plan.zed_holdout, 'path'])
    # Demo pool = members the notebook used ONLY for the held-out demo report.
    demo_members = set(plan.loc[plan.demo_group.notna(), 'path'])

    # The evaluation folder: member paths recorded by make_eval_subset.py.
    lab = json.load(open(a.labels))
    if 'members' not in lab: raise SystemExit('labels.json has no "members" field — it must come from make_eval_subset.py')
    eval_members = set(lab['members'])

    overlap_train = train_members & eval_members
    overlap_demo = demo_members & eval_members
    print(f'evaluation images: {len(eval_members)}')
    print(f'  used in TRAINING (must be 0): {len(overlap_train)}')
    print(f'  in the notebook demo pool (held out; overlap is harmless): {len(overlap_demo)}')
    print(f'  never seen by the notebook at all: {len(eval_members - train_members - demo_members)}')
    if overlap_train:
        print('LEAKAGE: these evaluation images were training images:'); [print('   ', m) for m in sorted(overlap_train)[:20]]
        raise SystemExit(1)
    print('OK — no evaluation image was used to fit any part of the model.')


if __name__ == '__main__':
    main()
