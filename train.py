"""Fine-tune Faster R-CNN on one split and score it with the competition metric.

    python train.py --split loso:ethz_1
    python train.py --split naive
    python train.py --split test        # the frozen test set: run once, at the end
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from wheat.annotations import load_marking
from wheat.dataset import WheatDataset, collate_batch
from wheat.metric import calculate_final_score
from wheat.model import build_model
from wheat.splits import load_splits


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def resolve_split(splits, name):
    if name == 'naive':
        return splits['naive']['train'], splits['naive']['val']
    if name == 'test':
        train = sorted(set(splits['naive']['train']) | set(splits['naive']['val']))
        return train, splits['test']
    if name.startswith('loso:'):
        fold = splits['loso'][name.split(':', 1)[1]]
        return fold['train'], fold['val']
    raise ValueError(f'unknown split {name!r}')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for images, targets, image_ids in loader:
        detections = model([im.to(device) for im in images])
        for det, tgt, image_id in zip(detections, targets, image_ids):
            out.append({
                'image_id': image_id,
                'gt_boxes': tgt['boxes'].numpy(),
                'pred_boxes': det['boxes'].cpu().numpy(),
                'scores': det['scores'].cpu().numpy(),
            })
    return out


def count_mae(predictions, score_threshold):
    errors = [abs(int((p['scores'] > score_threshold).sum()) - len(p['gt_boxes'])) for p in predictions]
    return float(np.mean(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data')
    parser.add_argument('--splits', default='splits/splits.json')
    parser.add_argument('--split', required=True, help='naive | test | loso:<source>')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--score-threshold', type=float, default=0.5)
    parser.add_argument('--max-train-images', type=int, default=None,
                        help='use only the first N training images (training-set-size study, smoke tests)')
    parser.add_argument('--max-val-images', type=int, default=None)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=5131)
    parser.add_argument('--out', default='runs')
    args = parser.parse_args()

    seed_everything(args.seed)
    device = pick_device(args.device)
    marking = load_marking(args.root)
    train_ids, val_ids = resolve_split(load_splits(args.splits), args.split)
    if args.max_train_images:
        train_ids = list(np.random.default_rng(args.seed).permutation(train_ids)[:args.max_train_images])
    if args.max_val_images:
        val_ids = val_ids[:args.max_val_images]

    train_loader = DataLoader(WheatDataset(args.root, marking, train_ids), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, collate_fn=collate_batch,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(WheatDataset(args.root, marking, val_ids), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, collate_fn=collate_batch)

    model = build_model(pretrained=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_dir = Path(args.out) / f"{args.split.replace(':', '_')}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for images, targets, _ in train_loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        predictions = predict(model, val_loader, device)
        row = {
            'epoch': epoch,
            'train_loss': float(np.mean(losses)),
            'val_map_0.5:0.75': calculate_final_score(predictions, args.score_threshold),
            'val_ap_0.5': calculate_final_score(predictions, args.score_threshold, thresholds=(0.5,)),
            'val_ap_0.75': calculate_final_score(predictions, args.score_threshold, thresholds=(0.75,)),
            'val_count_mae': count_mae(predictions, args.score_threshold),
        }
        history.append(row)
        print(json.dumps(row))
        torch.save(model.state_dict(), run_dir / 'model.pt')

    summary = {
        'args': vars(args),
        'device': str(device),
        'n_train': len(train_ids),
        'n_val': len(val_ids),
        'wall_clock_hours': (time.time() - start) / 3600,
        'history': history,
    }
    (run_dir / 'metrics.json').write_text(json.dumps(summary, indent=1))
    print('wrote', run_dir)


if __name__ == '__main__':
    main()
