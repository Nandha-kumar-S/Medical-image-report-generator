# Chest X-Ray Impression Classifier

Predicting the radiological *impression* for a chest X-ray study from its
frontal and lateral views, trained on the Indiana University chest X-ray
collection.

> **Scope note, up front:** the repository is named "report generator", but what
> is implemented is a **classifier**. It selects from impression strings seen
> during training — it does not generate new text. See
> [Honest limitations](#honest-limitations).

---

## The problem

Every chest X-ray study ends with a radiologist writing an impression — a short
summary of what the images show. At volume the task is repetitive, and fatigue
is a documented source of inconsistency between reads. Anything that drafts a
first pass, or flags when a study looks unlike the usual, saves attention for
the studies that need it.

## Dataset

The [Indiana University chest X-ray collection](https://openi.nlm.nih.gov/) —
paired radiograph images and their written reports.

| File | Contents |
|---|---|
| `indiana_reports.csv` | 3,851 reports — `indication`, `findings`, `impression`, MeSH terms |
| `indiana_projections.csv` | 7,466 image projections, labelled frontal / lateral |
| `merged_dataset.csv` | reports joined to their frontal and lateral filenames |
| `iu_dataset.csv` | the above **plus precomputed feature vectors** |

Most studies have two views, so each report maps to a frontal *and* a lateral
image.

## Approach

**1. Encode the images**

Each view is passed through **MobileNetV2** (ImageNet weights, `include_top=False`,
average pooling), producing a 1280-dimensional vector per view:

```python
self.base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
```

Frontal and lateral features are concatenated into a single **2560-d** vector —
using both views is the clinically sensible choice, since a lateral view resolves
things a frontal alone can be ambiguous about.

These vectors are precomputed and stored in `iu_dataset.csv`, so training doesn't
re-encode images on every run.

**2. Encode the target**

Impression strings are label-encoded into class IDs:

```python
label_encoder = LabelEncoder()
df['impressions_encoded'] = label_encoder.fit_transform(df['impression'])
```

**This is the decision that makes it a classifier rather than a generator** —
see below.

**3. Train**

A small Keras MLP over the concatenated features:

```python
Sequential([
    Dense(64, activation='relu', input_shape=input_shape),
    Dense(32, activation='relu'),
    Dense(output_shape, activation='softmax'),
])
```

An XGBoost variant (`models/xgb.py`) is also included, which additionally
TF-IDFs the impression text.

**4. Serve**

`app.py` exposes a Flask `/predict` endpoint. It encodes the uploaded frontal
and lateral images with the same MobileNetV2 encoder used for training,
concatenates them, and runs the MLP.

## Running it

```bash
pip install -r requirements.txt
python main.py                      # trains and writes saved_model/mlp_model.pkl
python app.py                       # serves on http://127.0.0.1:5000
```

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "frontal=@dataset/Images/1_IM-0001-3001.dcm.png" \
  -F "lateral=@dataset/Images/1_IM-0001-4001.dcm.png"
```

Both views are required — the model's input shape is frontal **and** lateral
concatenated.

## Honest limitations

**It classifies; it doesn't generate.** Label-encoding impressions and training a
softmax over those classes means the model can only ever return an impression it
has already seen. A genuine report generator needs a decoder — a seq2seq model,
or a language model conditioned on the image features. This is the single
biggest gap between the project's name and its implementation.

**Predictions come back as a class index, not text.** The `LabelEncoder` fitted
during training isn't persisted, so there's nothing to map the predicted index
back to its impression string. Saving the encoder alongside the model would fix
this in a few lines.

**Impression classes are heavily imbalanced.** "No acute cardiopulmonary
abnormality" and similar normal findings dominate the dataset. Plain accuracy is
therefore a poor metric here — a model that always predicts "normal" would score
well while being clinically useless. Per-class recall, or a macro-averaged F1,
would say much more.

**`encode_train_data()` in `main.py` does not run.** It's the offline step that
originally built the feature columns, and it depends on a `TextEncoder` class
that isn't in this repository. The features it produced are already in
`iu_dataset.csv`, so training doesn't need it. It's kept for reference and marked
as such.

**Not a clinical tool.** No validation against radiologist ground truth beyond
the held-out split, no calibration, no uncertainty estimates.

## What I'd change

- **Swap the softmax for a decoder** so the model can produce unseen text. This
  is the change that would make the project match its title.
- **Persist the `LabelEncoder`** next to the model so predictions return
  readable impressions.
- **Report per-class metrics**, not just accuracy, given the class imbalance.
- **Fine-tune the image encoder.** MobileNetV2's ImageNet weights are frozen;
  chest radiographs look nothing like ImageNet photographs, so even light
  fine-tuning would likely help a lot.
- **Group by patient when splitting.** Studies from the same patient appearing
  in both train and test would inflate results — worth verifying `uid` doesn't
  repeat across the split.

## Built with

Python · TensorFlow/Keras · MobileNetV2 · XGBoost · scikit-learn · Flask · pandas
