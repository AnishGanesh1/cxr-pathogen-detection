# Classification metrics

Consolidated model (parameter-space average of the partition checkpoints),
evaluated by `src/ensemble/evaluate_final.py` over the 498-image
patient-disjoint validation subset. Transcribed verbatim from
`evaluation_output.txt`.

**Micro F1 0.7528**  ·  **Exact match 0.00%**  ·  **Mean AUC-ROC 0.505**

| Pathology | Class accuracy | AUC-ROC |
|---|---:|---:|
| Atelectasis | 56.4% | 0.499 |
| Cardiomegaly | 62.4% | 0.474 |
| Consolidation | 68.3% | 0.447 |
| Edema | 66.7% | 0.509 |
| Enlarged Cardiomediastinum | 96.8% | 0.582 |
| Fracture | 53.6% | 0.505 |
| Lung Lesion | 86.7% | 0.487 |
| Lung Opacity | 32.5% | 0.471 |
| No Finding | 93.0% | 0.500 |
| Pleural Effusion | 95.6% | 0.566 |
| Pleural Other | 84.3% | 0.432 |
| Pneumonia | 60.8% | 0.527 |
| Pneumothorax | 92.8% | 0.522 |
| Support Devices | 64.5% | 0.554 |

## Reading these

Class accuracy and AUC-ROC pull in opposite directions here, and the gap is the
main finding. Enlarged Cardiomediastinum shows 96.8% accuracy on an AUC of
0.582: the class is rare, the model answers "absent" almost always, and it is
right almost always. Accuracy rewards that; AUC does not. Lung Opacity is the
mirror image, 32.5% accuracy because the class is common.

Every AUC sits between 0.432 and 0.582. Ranking is close to chance across all
fourteen classes, which is why the micro F1 of 0.7528 should not be read as
discriminative performance — it reflects the model getting the prevalent
negative cases right.

The paper reports a mean AUC of 0.759 with peaks at 0.87 over a different
14-class vocabulary. That figure is not reproduced by this evaluation. See
"Open discrepancies" in the top-level README.
