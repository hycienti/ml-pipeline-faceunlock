# FaceUnlock: a face-login ML pipeline

Register with a username and three photos of your face. Afterwards only your face unlocks that
username. Behind the login sits a dashboard that shows how the model performs, how long the service
has been up, what the data looks like, and lets anyone upload new faces and retrain the model.

| | |
|---|---|
| **Video demo** | _TODO: YouTube link (added after deployment)_ |
| **Live app** | _TODO: public URL (added after deployment)_ |
| **Repo** | https://github.com/hycienti/ml-pipeline-faceunlock |

## What it is

The model is a CNN (MobileNetV2 backbone, ImageNet-pretrained, fine-tuned by me) that turns a face
into a 128-number fingerprint. Two photos of the same person land close together, photos of different
people land far apart. The app compares fingerprints, never raw pixels.

* **Register**: 3 photos are averaged into one stored fingerprint.
* **Log in**: username + one photo. The photo's fingerprint must be close enough to the stored one.
* **Predict**: upload any image and the model names the registered person it matches, says "possible match", or says unknown. It uses a stricter bar than login that grows with the number of registered people.
* **Train and retrain**: upload a zip of new faces (one folder per person), then retrain from the model I trained (light or deep) or train a brand-new model from ImageNet weights.
* **Dashboard**: uptime, model metrics, ROC curve, dataset charts and what they mean, a live evaluation of
  the running model, and per-user insights (photo quality, where your face sits among others, look-alikes).

The task is classification on **image data**, not tabular data.

## What is in the project

| Capability | Where |
|---|---|
| Data acquisition | Notebook sections 1 to 4 (Kaggle via `kagglehub`) |
| Data processing | `src/preprocessing.py`, notebook sections 5 to 8 |
| Model creation / testing | `src/model.py`, `src/training.py`, `src/evaluation.py`, notebook sections 9 to 13 |
| Retraining + trigger | `src/retrain.py`, `app/retrain_routes.py`, dashboard "Upload data and retrain" |
| API with Python | Flask app in `app/` (`/api/predict`, `/api/login`, `/api/datasets`, `/api/retrain`, ...) |
| UI: uptime, visualisations, train/retrain | Dashboard (`app/templates/dashboard.html`) |
| Prediction of one data point | Dashboard "Predict: who is this?" and `POST /api/predict` |
| Visualisations with interpretation of 3+ features | Dashboard "What story does the data tell?" and notebook section 8c |
| Bulk upload for retraining | Dashboard "Upload data and retrain" (zip of many images) |
| Evaluation in production | Dashboard "Evaluation in production" (`/api/evaluate`) |
| Flood simulation (Locust) | See "Flood simulation" below |
| Model file | `models/faceunlock_v1.keras` and `models/faceunlock_v1.h5` |

## Data

* **LFW (Labeled Faces in the Wild)**: 5,749 people, 13,233 photos. Used for training and evaluation
  because it has real identities. Split **by person** (830 train / 118 validation / 236 test people),
  with LFW's official test people held out entirely for the official benchmark.
* **Kaggle "Human Faces"**: about 7,200 unlabeled portraits. Used for exploration, the dashboard charts
  and the face map, not for training (it has no identities).

Findings that shaped the pipeline (full numbers on the dashboard and in notebook section 8c): 71% of LFW
people have one photo, so only 1,680 are usable and each person is capped at 20 photos; the Kaggle photos
range from 236 to 7,680 pixels wide, so every face is cropped and resized to 160x160; brightness varies from
very dark to washed out, so training augments brightness and contrast; the detector finds exactly one face in
about 92% of photos, so the app rejects photos without a single face.

## Results

Evaluation is **verification**: given two photos, are they the same person? Measured on people the model
never saw during training. Training used a cumulative set of techniques (each row adds to the previous one):

| Run | Test AUC | LFW official AUC |
|---|---|---|
| 1 baseline (frozen backbone) | 0.844 | 0.823 |
| 2 + augmentation | 0.827 | 0.843 |
| 3 + regularisation (dropout, weight decay, early stopping) | 0.829 | 0.852 |
| 4 + fine-tuning the whole backbone, cosine decay | **0.887** | **0.888** |

Final model, at the validation-chosen equal-error threshold:

| Metric | My held-out test people | LFW official pairs |
|---|---|---|
| Accuracy | 0.808 | 0.806 |
| Precision | 0.798 | 0.802 |
| Recall | 0.825 | 0.812 |
| F1 | 0.811 | 0.807 |
| AUC | 0.887 | 0.888 |

Optimisation: Adam with gradient clipping, a two-phase schedule (head first, then backbone at a smaller
learning rate), a scaled cosine-margin loss (CosFace style), ReduceLROnPlateau, cosine decay and early
stopping on validation AUC. The notebook records three attempts honestly, including the first one where
a plain softmax head underfit and every technique made the model worse (section 10).

**What this means in practice.** The app uses a strict login threshold (about 2% of strangers accepted on
validation pairs). Measured the way the app is used, over 66 held-out people enrolled from 3 photos: the right
person is accepted on the first try **67%** of the time and strangers **3.9%** of the time (77% for the right
person if you accept 5% strangers). That is usable for a demo, not for real security. See Limitations.

## Repository layout

```
README.md
notebook/faceunlock.ipynb      EDA, preprocessing, training, evaluation, prediction demo (runs on Colab)
src/preprocessing.py           face detection, cropping, resizing, pair building
src/model.py                   embedding CNN, cosine-margin head, augmentation
src/prediction.py              FaceVerifier: embed, enroll, verify, compare
src/training.py  evaluation.py retrain.py  insights.py  ...   training loop, metrics, retraining, dashboard insights
app/                           Flask app: auth, dashboard APIs, retraining, templates, static files
data/train  data/val  data/test    person-level splits (filled by scripts/prepare_data.py)
models/                        faceunlock_v1.keras / .h5, metrics_v1.json, replay_faces.zip, dashboard data
scripts/                       prepare_data, export_h5, set_login_threshold, reembed_users, make_replay_zip
tests/                         pytest suites
Dockerfile  compose.yaml  nginx.conf  modal_app.py      deployment
```

## Set it up

### 1. Run it locally

Needs Python 3.11 or 3.12 (TensorFlow does not support newer versions yet).

```bash
git clone https://github.com/hycienti/ml-pipeline-faceunlock.git
cd ml-pipeline-faceunlock
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
flask --app app:create_app run --port 8000
```

Open http://127.0.0.1:8000/register (the camera works on `127.0.0.1` without HTTPS; every page also has a
file-upload fallback). The trained model is in `models/`, so nothing else is needed to run the app.

### 2. Get the data (only to retrain from scratch or rerun the notebook)

```bash
pip install -r requirements-train.txt
python -c "import kagglehub; print(kagglehub.dataset_download('jessicali9530/lfw-dataset'))"
# copy or unzip the result into data/lfw so that data/lfw/lfw-deepfunneled/... exists
python scripts/prepare_data.py          # creates data/train, data/val, data/test
```

### 3. Run the notebook (training happens here, on a GPU)

Upload `notebook/faceunlock.ipynb` to Google Colab, set **Runtime > Change runtime type > GPU**, and run all
cells. It clones this repo, downloads both datasets from Kaggle itself, trains, evaluates, and downloads
`faceunlock_v1.keras`, `faceunlock_v1.h5` and `metrics_v1.json`. Put them in `models/`, then:

```bash
python scripts/set_login_threshold.py   # adds the login threshold (2% strangers) to metrics_v1.json
python scripts/reembed_users.py         # only if people were already registered: re-enrol them with the new model
```

### 4. Docker, and several containers behind a load balancer

```bash
docker compose up --build --scale web=3      # 3 app containers behind nginx
# open http://localhost:8080
```

Containers share one database volume (SQLite in WAL mode) and one session key, so scaling needs no other setup. nginx
looks up the containers when it starts, so after changing the number run `docker compose up -d --scale web=N` and then
`docker compose restart proxy`. The proxy waits until every container is healthy, and balances by least connections.

### 5. Deploy on Modal

```bash
pip install modal && python3 -m modal setup     # one-time browser login
modal deploy modal_app.py                       # prints the public https URL
```

One container (SQLite has a single writer) that sleeps when idle; the dashboard counts the sleep as "up".
Users, photos and models live on a Modal Volume.

### 6. Tests

```bash
python -m pytest tests -q -k "not retrain"      # fast suites; the retrain tests train tiny real models
```

## Using the app

1. **Register**: username + 3 photos (camera or upload). Photos without exactly one face are rejected.
2. **Log in**: username + photo. Five failed attempts lock that username for five minutes. A photo with no face
   is not counted as a failed attempt.
3. **Dashboard**: service status and uptime graph; **Predict** (upload or camera: who is this?); **About your face**
   (photo quality, face map, look-alikes, login margin); model metrics and ROC; **Evaluation in production**
   (press the button to score the running model on held-out pairs; every run is saved); the data and what it says.
4. **Upload data and retrain**:
   * Upload a zip: `person_name/photo.jpg`, at least 10 people with 2+ photos each. It is validated (corrupt zips,
     bad images and path tricks are rejected), preprocessed (the report shows how many photos have a detectable face),
     and saved to the database.
   * **Start** trains on every saved dataset, in one of three modes. **Light** retrains only the top layer (minutes).
     **Deep** also fine-tunes the whole backbone (slow on a CPU). Both start from the model I trained and mix in a sample
     of the original people so it does not forget them. **Train from scratch** ignores the current model and trains a new one
     from ImageNet weights on the upload plus every original photo, the same recipe as the notebook (very slow on a CPU).
   * The dashboard shows old vs new on the uploaded people and on the original validation pairs ("did it forget?").
     Nothing goes live until you press **Promote**, which re-enrols every user with the new model.
   * **Trigger when the need arises**: once 10 new people are waiting the dashboard says retraining is recommended,
     and setting the environment variable `AUTO_RETRAIN_PEOPLE=10` (or any number) makes a light retrain start automatically at that point. `RETRAIN_RECOMMEND_PEOPLE` changes when the recommendation appears.

## Flood simulation (Locust)

_TODO: results with 1, 2 and 3 containers (latency and response time), added after deployment._

## Limitations (please read)

* **The model is modest.** AUC 0.89 and about 67% first-try acceptance for the right person at the strict
  threshold. A face-pretrained backbone or far more identities would be needed to do clearly better. The
  notebook documents the three training attempts and why more fine-tuning did not help.
* **Identification is weaker than login.** `/api/predict` compares a photo with every registered person, so a stranger gets
  one chance per registered person. At the plain login bar a stranger was wrongly named 40% of the time against a 15-person
  gallery. The predict bar therefore rises with the number of registered people; the price is that fewer registered people are
  named (about 75% with 1 registered person, 61% with 4, 47% with 15, while strangers are wrongly named 6% to 14% of the time;
  notebook section 13). Faces between the two bars are reported as "possible match", with the closest candidates and scores.
* **No liveness check**: a printed photo of you would work. Not defended against.
* **Enrollment photos are stored** on the server so a new model can re-enrol users without asking them to register again.
* **SQLite** means one writer: run more containers for read-heavy load, not for heavy registration traffic.
* A retrain on a small upload is noisy; compare old vs new before promoting.
