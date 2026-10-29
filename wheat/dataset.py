# Adapted from wuxinwang1997/wheatdetection data/datasets/train_wheat.py (MIT, (c) 2020 CSRobin).
# Changed: removed augmentations; returns a CHW tensor; allows images with no boxes.
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class WheatDataset(Dataset):

    def __init__(self, root, marking, image_ids):
        super().__init__()
        self.root = root
        self.image_ids = np.asarray(image_ids)
        self.marking = marking

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image, boxes = self.load_image_and_boxes(index)

        # there is only one class
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])

        image = torch.from_numpy(image).permute(2, 0, 1)
        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
        image = cv2.imread(f'{self.root}/train/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.marking[self.marking['image_id'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values.copy()
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return image, boxes


def collate_batch(batch):
    return tuple(zip(*batch))
