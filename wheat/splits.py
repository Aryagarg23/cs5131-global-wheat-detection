"""Source-based split protocol from the CS5131 interim checkpoint, Section 2.

Frozen test set: two whole sources, picked with a fixed seed, holding 15-25% of the images
and at least one source from outside France. Development set: the other five sources, used
for leave-one-source-out folds. Naive split: random 80/20 image-level split of the
development set, ignoring source, to measure leakage.

Images with no boxes have no source label and are left out of every split.
"""
import itertools
import json
import random

import numpy as np

FRANCE_SOURCES = {'arvalis_1', 'arvalis_2', 'arvalis_3', 'inrae_1'}
TEST_FRACTION_RANGE = (0.15, 0.25)
SEED = 5131


def labelled(table):
    return table[table['source'].notna()]


def pick_test_sources(table, seed=SEED):
    counts = labelled(table)['source'].value_counts()
    total = counts.sum()
    pairs = sorted(itertools.combinations(sorted(counts.index), 2))
    random.Random(seed).shuffle(pairs)
    lo, hi = TEST_FRACTION_RANGE
    for pair in pairs:
        frac = counts[list(pair)].sum() / total
        if lo <= frac <= hi and not set(pair) <= FRANCE_SOURCES:
            return sorted(pair)
    raise ValueError('no pair of sources satisfies the test-set constraints')


def test_and_dev(table, test_sources):
    lab = labelled(table)
    is_test = lab['source'].isin(test_sources)
    return lab[is_test]['image_id'].to_numpy(), lab[~is_test]


def loso_folds(dev):
    """{held-out source: (train image ids, val image ids)} over the development sources."""
    folds = {}
    for source in sorted(dev['source'].unique()):
        held = dev['source'] == source
        folds[source] = (dev[~held]['image_id'].to_numpy(), dev[held]['image_id'].to_numpy())
    return folds


def naive_split(dev, val_fraction=0.2, seed=SEED):
    ids = np.array(sorted(dev['image_id']))
    np.random.default_rng(seed).shuffle(ids)
    n_val = int(round(len(ids) * val_fraction))
    return ids[n_val:], ids[:n_val]


def build_splits(table, seed=SEED):
    test_sources = pick_test_sources(table, seed)
    test_ids, dev = test_and_dev(table, test_sources)
    naive_train, naive_val = naive_split(dev, seed=seed)
    return {
        'seed': seed,
        'test_sources': test_sources,
        'test': sorted(test_ids.tolist()),
        'loso': {s: {'train': sorted(tr.tolist()), 'val': sorted(va.tolist())}
                 for s, (tr, va) in loso_folds(dev).items()},
        'naive': {'train': sorted(naive_train.tolist()), 'val': sorted(naive_val.tolist())},
        'excluded_no_boxes': sorted(table[table['source'].isna()]['image_id'].tolist()),
    }


def write_splits(splits, path):
    with open(path, 'w') as f:
        json.dump(splits, f, indent=1)


def load_splits(path):
    with open(path) as f:
        return json.load(f)


if __name__ == '__main__':
    import argparse

    from wheat.annotations import load_image_table

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data')
    parser.add_argument('--out', default='splits/splits.json')
    args = parser.parse_args()

    table = load_image_table(args.root)
    splits = build_splits(table)
    write_splits(splits, args.out)
    lab = labelled(table)
    print('test sources:', splits['test_sources'],
          f"({len(splits['test'])} images, {len(splits['test']) / len(lab):.1%} of labelled)")
    for s, f in splits['loso'].items():
        print(f"loso {s:10s} train={len(f['train'])} val={len(f['val'])}")
    print(f"naive train={len(splits['naive']['train'])} val={len(splits['naive']['val'])}")
    print('excluded (no boxes, no source):', len(splits['excluded_no_boxes']))
