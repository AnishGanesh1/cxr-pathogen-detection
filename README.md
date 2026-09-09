# A Dual-Task Deep Learning Framework for Multi-Label Thoracic Pathology Classification and Spatial Localization in Chest Radiographs with Grad-CAM Explainability

LaTeX source, figures, and the project's training and evaluation code.

**Authors** — Allan Suraj, Anish Ganesh, Arpan Murthy, Kshitij Narendrakumar
Nagashetti, Rashmi N Ugarakhod
Department of Electronics and Communication Engineering, PES University, Bangalore, India

## What this is

A dual-task framework for chest radiograph interpretation:

- **Classification** — DenseNet-121 trained under binary Focal Loss
  (alpha = 0.25, gamma = 2.0) on MIMIC-CXR, 14 CheXpert-derived pathologies
- **Localization** — Faster R-CNN over an FPN neck on the radiologist-annotated
  VinDr-CXR boxes
- **Explainability** — Grad-CAM saliency from the final dense block, fused with
  detector boxes at inference
- **Deployment** — Streamlit application (MED-AI Visualizer)

## Layout

```
paper.tex            IEEEtran conference source (6 pages)
paper_anon.tex       identical, author names and affiliations withheld
paper.pdf            compiled paper
paper_anon.pdf       compiled anonymous version
fig/                 figures used by the paper
references.bib       bibliography
src/                 training and evaluation code
results/             evaluation output as produced by the scripts
scripts/             figure generation and analysis
docs/                submission notes, roadmap, similarity analysis
```

## Code

```
src/classification/   DenseNet-121 multi-label classifier
    focal_loss.py         Focal Loss, alpha=0.25 gamma=2.0
    mimic_dataset.py      MIMIC-CXR loader (reads a labels CSV; images not included)
    model_mimic.py        backbone + 1024->512->14 sigmoid head
    train_mimic.py        training loop, Adam + ReduceLROnPlateau + AMP
    evaluate_mimic.py     micro/macro F1, per-class AUC, exact match
    predict_single_image.py, predict_with_heatmap.py   inference and Grad-CAM

src/partitions/       per-partition training, p13 through p19
src/ensemble/         consolidate_models.py  uniform weight averaging in parameter space
src/detection/        VinDr-CXR: csv_to_coco.py converts per-reader CSV to COCO JSON
src/metrics/          bootstrap confidence intervals
```

## Building the paper

Requires a TeX distribution with IEEEtran (texlive-publishers on Debian/Ubuntu),
plus algorithmic, booktabs and tikz.

```bash
make          # builds paper.pdf
make anon     # builds paper_anon.pdf
make clean
```

Both variants must come out at exactly 6 pages, the venue limit.

## Data

Not redistributed here, and not redistributable. Both corpora are
credentialed-access through PhysioNet, and their data use agreements prohibit
sharing the images or the derived label files:

- MIMIC-CXR   https://physionet.org/content/mimic-cxr/
- VinDr-CXR   https://physionet.org/content/vindr-cxr/

mimic_dataset.py expects a labels CSV and an image root; point them at your own
credentialed copy. Trained checkpoints are likewise not included, and .pth,
.csv and .dcm are gitignored so they cannot be committed by accident.

## Open discrepancies

Three things to check before relying on the numbers in the paper.

**1. Per-class AUC.** results/evaluation_results.txt, produced by
src/ensemble/evaluate_final.py over the 498-image validation subset, reports
per-class AUC-ROC between 0.432 and 0.582, mean about 0.505, over the CheXpert
14 vocabulary. The paper reports a mean of 0.759 with peaks at 0.87, over a
different 14-class vocabulary. The micro F1 (0.7528) and exact match (0.00%) do
agree between the two. The source of the AUC figures in the paper has not been
located in this repository.

**2. scripts/generate_figures.py does not plot measured results.** Its ROC
routine seeds a random generator and draws curves to hard-coded AUC targets
(aucs = [0.89, 0.86, 0.87, 0.78, 0.81, 0.85]), and its class-distribution
routine uses hard-coded counts described in the source as "representative"
rather than counted from the corpus. Its output is illustrative, not
evidential, and must not be presented as a result.

**3. Confusion matrix.** The diagonal of Fig. 3 sums to 130 while the text and
Table II report top-1 agreement as 129/487 (26.5%).
