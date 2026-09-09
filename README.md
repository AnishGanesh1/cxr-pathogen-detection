# Dual-Task Chest Radiograph Interpretation

Multi-label pathology classification on MIMIC-CXR, lesion localization on
VinDr-CXR, and Grad-CAM evidence for both — with the training code, the
evaluation output and the qualitative results that came out of it.

**Authors** — Allan Suraj, Anish Ganesh, Arpan Murthy, Kshitij Narendrakumar
Nagashetti, Rashmi N Ugarakhod
Department of Electronics and Communication Engineering, PES University, Bangalore, India

---

## The pipeline

Two branches, trained separately on two corpora with disjoint label taxonomies,
joined at inference behind one interface.

### 1. Data preparation

MIMIC-CXR ships as patient-prefix folders with NLP-derived labels in the
CheXpert vocabulary. VinDr-CXR ships as one CSV row per radiologist per finding.
Neither is usable as-is.

- `src/detection/csv_to_coco.py` — collapses the per-reader VinDr rows into
  COCO-style JSON, rewriting each box as `[xmin, ymin, w, h]` and reserving an
  index for background
- `src/classification/mimic_dataset.py` — reads the MIMIC labels CSV and image
  root, resizes to 256×256, replicates the single grey channel three times so
  ImageNet weights load without touching the stem convolution
- Splits are cut at the patient-prefix level, so no patient can appear in both
  training and validation

Augmentation is deliberately weak — horizontal flip and `ColorJitter(0.1, 0.1)`
only — because absolute grey level on a radiograph is itself diagnostic.

### 2. Classification branch

DenseNet-121, ImageNet-initialised, 1000-way head replaced by
Linear(1024→512) → ReLU → Dropout(0.2) → Linear(512→14) → Sigmoid.

Fourteen independent sigmoids rather than a softmax, because one film routinely
carries several findings at once. Trained under Focal Loss
(`src/classification/focal_loss.py`, alpha 0.25, gamma 2.0) to stop the abundant
easy negatives from dominating the gradient, with Adam, ReduceLROnPlateau and
mixed precision.

Each partition p13–p18 was trained on its own (`src/partitions/train_p*.py`),
producing one checkpoint per patient population.

### 3. Ensembling

`src/ensemble/consolidate_models.py` averages those checkpoints **in parameter
space** — load each `state_dict`, stack matching keys, take the mean. Unlike
averaging outputs, this costs nothing at inference, which matters under a
real-time target.

### 4. Localization branch

Faster R-CNN over an FPN neck (`src/detection/vindr.py`), trained on the
radiologist-drawn VinDr boxes. Pyramid levels P2–P6 feed a proposal network;
survivors are pooled by RoIAlign and passed to paired classification and
box-regression heads. Boxes are drawn above an operator-set threshold, 0.30 by
default.

### 5. Explainability

`src/classification/predict_with_heatmap.py` takes gradients of the class logit
with respect to the final dense block, averages per channel, forms the rectified
weighted sum, upsamples and alpha-blends over the film. Heatmaps are drawn only
for predictions above a 30% display threshold, so a reader is never shown
evidence for a finding the model did not assert.

---

## Results

See **[`results/`](results/)** for the full set. Headline, on the 498-image
patient-disjoint validation subset:

| Metric | Value |
|---|---|
| Micro F1 | 0.7528 |
| Exact match ratio | 0.00% |
| Mean AUC-ROC | 0.505 |

- [`results/classification_metrics.md`](results/classification_metrics.md) — per-class accuracy and AUC for all 14 classes, with a note on why the two disagree
- [`results/evaluation_output.txt`](results/evaluation_output.txt) — raw run log
- [`results/inference_result.txt`](results/inference_result.txt) — single-study inference trace
- [`results/qualitative/mimic_gradcam/`](results/qualitative/mimic_gradcam/) — six Grad-CAM overlays, one per partition checkpoint
- [`results/qualitative/vindr_detection/`](results/qualitative/vindr_detection/) — Faster R-CNN boxes on VinDr test films

The Grad-CAM set is one overlay per partition on the same study, so the
partition-to-partition variation is directly visible.

---

## Layout

```
src/classification/   classifier, Focal Loss, MIMIC loader, Grad-CAM inference
src/partitions/       per-partition training, p13 through p18
src/ensemble/         parameter-space weight averaging + final evaluation
src/detection/        VinDr COCO conversion, Faster R-CNN
src/metrics/          bootstrap confidence intervals
results/              metrics, logs and qualitative outputs
scripts/              figure generation for the paper
fig/                  figures as they appear in the paper
paper.tex             IEEEtran conference source (6 pages)
paper_anon.tex        same, author names and affiliations withheld
references.bib
```

---

## Data

**Not redistributed here, and not redistributable.** Both corpora are
credentialed-access through PhysioNet, and their data use agreements prohibit
sharing the images or the derived label files:

- MIMIC-CXR — https://physionet.org/content/mimic-cxr/
- VinDr-CXR — https://physionet.org/content/vindr-cxr/

Point `mimic_dataset.py` at your own credentialed copy. Trained checkpoints are
not included. `.pth`, `.csv` and `.dcm` are gitignored so data and weights
cannot be committed by accident.

The images under `results/qualitative/` are radiographs from these corpora with
model output drawn on them. A small illustrative set is kept, the way a paper
figure would; the full prediction output is not included. **Confirm this is
acceptable under your DUA before making this repository public.**

---

## Building the paper

Requires a TeX distribution with IEEEtran (`texlive-publishers` on
Debian/Ubuntu), plus `algorithmic`, `booktabs` and `tikz`.

```bash
make          # paper.pdf
make anon     # paper_anon.pdf
make clean
```

Both come out at exactly 6 pages, the venue limit.

---

## Open discrepancies

Three things to check before relying on the numbers in the paper.

**1. Per-class AUC.** `results/evaluation_output.txt` reports per-class AUC-ROC
between 0.432 and 0.582, mean about 0.505, over the CheXpert 14 vocabulary. The
paper reports a mean of 0.759 with peaks at 0.87, over a different 14-class
vocabulary. Micro F1 (0.7528) and exact match (0.00%) do agree. The source of
the paper's AUC figures has not been located in this repository.

**2. `scripts/generate_figures.py` does not plot measured results.** Its ROC
routine seeds a random generator and draws curves to hard-coded AUC targets
(`aucs = [0.89, 0.86, 0.87, 0.78, 0.81, 0.85]`); its class-distribution routine
uses counts the source itself calls "representative" rather than counted from
the corpus. Its output is illustrative, not evidential.

**3. Confusion matrix.** The diagonal of Fig. 3 sums to 130 while the text and
Table II report top-1 agreement as 129/487 (26.5%).
