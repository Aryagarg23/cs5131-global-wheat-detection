# Global Wheat Detection (CS5131)

Find every wheat head in a photo of a wheat field. Data is from Kaggle's
[Global Wheat Detection](https://www.kaggle.com/competitions/global-wheat-detection) competition.

The main question is whether a detector still works on fields it has never seen. So we
split the data by source (which farm or country it came from), not by random image.

## Run it

You need Python 3 and a Kaggle account.

1. Accept the competition rules: https://www.kaggle.com/competitions/global-wheat-detection/rules
2. Make an API token at https://www.kaggle.com/settings and save it to `~/.kaggle/access_token`
3. Run:

```bash
./setup.sh
```

This sets up a Python env, downloads the data (about 600 MB) and runs the tests.

Then train:

```bash
.venv/bin/python train.py --split loso:ethz_1
```

`--split` can be:
- `loso:<source>` trains on the other dev sources and tests on that one (`arvalis_1`, `arvalis_2`, `ethz_1`, `inrae_1`, `rres_1`)
- `naive` is a random 80/20 split, used for comparison
- `test` is the held-out test set. Only run this once, at the end

Results go in `runs/`. For a quick check that everything works, add
`--epochs 1 --max-train-images 16 --max-val-images 8`.

## Splits

These are fixed in `splits/splits.json` (seed 5131).

- Test set: `arvalis_3` and `usask_1`, 759 images (22.5%).
- The other 5 sources are for training and leave-one-source-out folds.
- 49 images have no boxes, so they have no source label. They are left out of every split.

## Code

- `wheat/annotations.py` reads the boxes from `train.csv`
- `wheat/splits.py` makes the splits
- `wheat/dataset.py` is the PyTorch dataset
- `wheat/metric.py` is the Kaggle score (mean precision over IoU 0.5 to 0.75)
- `wheat/model.py` is Faster R-CNN from torchvision
- `train.py` trains and scores the model

Some of this is adapted from other people's code. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The data itself isn't in this repo. Kaggle's rules don't allow sharing it.
