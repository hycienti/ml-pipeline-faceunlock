# data/

The photos are not stored in git (they are large and come from public datasets). This folder is
filled by two commands, and `train/`, `val/` and `test/` are the splits the model is trained and
evaluated on.

| Folder | What it holds | How it appears |
|---|---|---|
| `lfw/` | Labeled Faces in the Wild (LFW), extracted from the Kaggle download | `kagglehub.dataset_download("jessicali9530/lfw-dataset")` or unzip the Kaggle archive here |
| `human-faces/` | Kaggle "Human Faces": ~7,200 unlabeled portraits, used for exploration and the dashboard | `kagglehub.dataset_download("ashwingupta3012/human-faces")` |
| `train/`, `val/`, `test/` | `<person>/<photo>.jpg` for each split (830 / 118 / 236 people) | `python scripts/prepare_data.py` |
| `cache/` | Pre-cropped face arrays created by the notebook | notebook section 8 |

The split is by **person**: nobody in `val/` or `test/` appears in `train/`, and LFW's official test
people are excluded from all three (they are kept untouched for the official benchmark).
`models/replay_faces.zip` holds the same training and validation photos so the deployed app can
mix them into a retrain.
