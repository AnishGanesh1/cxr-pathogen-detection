# Results

Everything here was produced by the code in `src/`. Nothing is hand-authored or
simulated.

## Files

| Path | Produced by | What it is |
|---|---|---|
| `classification_metrics.md` | `src/ensemble/evaluate_final.py` | per-class table, transcribed from the run log |
| `evaluation_output.txt` | `src/ensemble/evaluate_final.py` | raw console log of that run |
| `inference_result.txt` | `src/classification/predict_single_image.py` | 14 class probabilities for one p19 study |
| `qualitative/mimic_gradcam/` | `src/classification/predict_with_heatmap.py` | Grad-CAM overlays, one per partition checkpoint |
| `qualitative/vindr_detection/` | `src/detection/vindr.py` | Faster R-CNN boxes drawn on VinDr test films |

## How these were generated

**Classification.** DenseNet-121, ImageNet-initialised, 1000-way head replaced by
Linear(1024→512) → ReLU → Dropout(0.2) → Linear(512→14) → Sigmoid. Trained under
Focal Loss (alpha 0.25, gamma 2.0) with Adam, ReduceLROnPlateau, and AMP at
256×256. Each MIMIC-CXR patient-prefix partition p13–p18 was trained separately
(`src/partitions/train_p*.py`) so no patient appears in both train and
validation.

**Ensembling.** `src/ensemble/consolidate_models.py` averages the partition
checkpoints in parameter space — it loads each `state_dict`, stacks matching
keys and takes the mean. Unlike output-space ensembling this costs nothing at
inference. `evaluate_final.py` then scores the averaged model over the
498-image held-out subset.

**Grad-CAM.** `predict_with_heatmap.py` takes gradients of the class logit with
respect to the final dense block, averages them per channel, forms the rectified
weighted sum, upsamples and alpha-blends it over the radiograph. One overlay per
partition checkpoint is in `qualitative/mimic_gradcam/`, which is what makes the
partition-to-partition variation visible.

**Detection.** `src/detection/csv_to_coco.py` converts the VinDr per-radiologist
CSV annotations into COCO JSON, boxes as `[xmin, ymin, w, h]` with a reserved
background index. `vindr.py` trains and runs Faster R-CNN over an FPN neck and
draws the surviving boxes above the 0.30 operating point.

## Headline numbers

498-image patient-disjoint validation subset, consolidated model:

- Micro F1 **0.7528**
- Exact match ratio **0.00%** (expected: 14 independent binary decisions must all be right)
- Mean AUC-ROC **0.505** across the 14 CheXpert classes

## A note on the images

The overlays and detection visualisations are MIMIC-CXR and VinDr-CXR
radiographs with model output drawn on top. They are derived from
credentialed-access data. A small illustrative set is kept here; the full
prediction output is not redistributed. See the Data section of the top-level
README before making this repository public.
