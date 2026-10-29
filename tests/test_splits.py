from pathlib import Path

import pytest

from wheat.annotations import load_image_table
from wheat.splits import FRANCE_SOURCES, build_splits, load_splits

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / 'data/train.csv').exists(), reason='no data')


@pytest.fixture(scope='module')
def table():
    return load_image_table(ROOT / 'data')


def test_download_complete(table):
    assert len(table) == 3422
    assert table['n_boxes'].sum() == 147793


def test_splits(table):
    s = build_splits(table)
    assert s == load_splits(ROOT / 'splits/splits.json')

    src = table.set_index('image_id')['source']
    dev = set(s['naive']['train']) | set(s['naive']['val'])
    assert not set(s['test']) & dev
    assert 0.15 <= len(s['test']) / src.notna().sum() <= 0.25
    assert not set(s['test_sources']) <= FRANCE_SOURCES

    for held, fold in s['loso'].items():
        assert set(src[fold['val']]) == {held}
        assert held not in set(src[fold['train']])
