from pathlib import Path

import pytest

from wheat.annotations import load_marking
from wheat.dataset import WheatDataset

ROOT = Path(__file__).resolve().parents[1] / 'data'
pytestmark = pytest.mark.skipif(not (ROOT / 'train.csv').exists(), reason='no data')


def test_item():
    marking = load_marking(ROOT)
    image_id = marking['image_id'].iloc[0]
    image, target, _ = WheatDataset(ROOT, marking, [image_id])[0]
    boxes = target['boxes']
    assert image.shape == (3, 1024, 1024)
    assert len(boxes) == (marking['image_id'] == image_id).sum()
    assert (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all()
