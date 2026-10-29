# load_marking adapted from wuxinwang1997/wheatdetection data/build.py (MIT, (c) 2020 CSRobin).
# Changed: removed its k-fold split; added load_image_table.
from pathlib import Path

import numpy as np
import pandas as pd

# Boxes that cover large parts of the image are annotation errors. The area cut-off and the
# list of exact bad areas are from the original code.
MAX_BOX_AREA = 154200.0
ERROR_BOX_AREAS = [100648.0, 145360.0, 149744.0, 119790.0, 106743.0]


def load_marking(root):
    marking = pd.read_csv(f'{root}/train.csv')

    bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        marking[column] = bboxs[:, i]
    marking = marking.drop(columns=['bbox'])
    marking['area'] = marking['w'] * marking['h']
    marking = marking[marking['area'] < MAX_BOX_AREA]
    marking = marking[~marking['area'].isin(ERROR_BOX_AREAS)]
    return marking.reset_index(drop=True)


def load_image_table(root):
    """One row per image on disk: image_id, source, n_boxes.

    train.csv only has rows for boxes, so images with no boxes have source NaN.
    """
    raw = pd.read_csv(f'{root}/train.csv')
    on_disk = sorted(p.stem for p in Path(root, 'train').glob('*.jpg'))
    table = pd.DataFrame({'image_id': on_disk})
    source = raw.groupby('image_id')['source'].first()
    n_boxes = raw.groupby('image_id').size()
    table['source'] = table['image_id'].map(source)
    table['n_boxes'] = table['image_id'].map(n_boxes).fillna(0).astype(int)
    return table
