# IEEE Paper — Compilation Guide, Fill-In Checklist & Source Discrepancies

## 1. What you have

| File | Purpose |
|---|---|
| `cxr_ieee_paper.tex` | LaTeX source, `IEEEtran` **conference** class (the official IEEE template class) |
| `cxr_ieee_paper.pdf` | Compiled 10-page two-column paper |

**To compile** (Overleaf: upload the `.tex`, set compiler to pdfLaTeX):
```bash
pdflatex cxr_ieee_paper.tex
pdflatex cxr_ieee_paper.tex   # second pass resolves cross-references
```
For a **journal** submission (JBHI, Computers in Biology and Medicine) change
line 1 to `\documentclass[journal]{IEEEtran}` and add author biographies +
photos at the end.

---

## 2. IEEE required elements — status

| Required element | In the paper? | Where |
|---|---|---|
| Title (concise, no abbreviations) | ✅ | top |
| Author block: name, dept., institution, city, country, email | ⚠️ placeholder | `%% <<< EDIT` |
| Abstract, 150–250 words, single paragraph, no citations | ✅ (~240 w) | after title |
| Index Terms (alphabetical-ish, 5–12) | ✅ | after abstract |
| Introduction with explicit contributions list | ✅ | §I |
| Related Work / literature survey | ✅ | §II |
| Materials & Methods with reproducible detail | ✅ | §III–IV |
| All equations numbered and referenced in text | ✅ | (1)–(13) |
| Algorithm block | ✅ | Algorithm 1 |
| Experimental setup + hyperparameter table | ✅ | §V, Table II |
| Metric definitions (not just names) | ✅ | §V-B |
| Results with tables **and** figures | ⚠️ tables need numbers | §VI |
| Ablation study | ⚠️ protocol defined, numbers needed | Table VI |
| Discussion of negative/anomalous results | ✅ | §VI-A |
| Limitations | ✅ | §VII |
| Conclusion + Future Work | ✅ | §VIII |
| **Ethics / data-availability statement** (mandatory for medical data) | ✅ | unnumbered section |
| Acknowledgment (funding, grant numbers) | ⚠️ placeholder | unnumbered section |
| References in IEEE numeric style `[1]`, cited in order | ✅ | 29 refs |
| Figure captions **below**, table captions **above** | ✅ | throughout |
| Page numbers off, no headers/footers | ✅ | handled by IEEEtran |

---

## 3. Things you must fill in before submitting

Search the `.tex` for `%% <<< EDIT` — there are 8 of them.

1. **Authors** — names, departments, institution, city/country, emails. Add
   `\thanks{}` for supervisor/funding if needed.
2. **Table II** — batch size, number of epochs, and hardware (GPU model + VRAM,
   CPU, RAM). Reviewers reject papers without hardware specs.
3. **Table IV (per-class results)** — 14 rows × precision/recall/F1/AUC +
   prevalence. Get this from `evaluate_mimic.py`, or:
   ```python
   from sklearn.metrics import classification_report, roc_auc_score
   print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))
   print(roc_auc_score(y_true, y_score, average=None))
   ```
   This table is the single most important missing piece.
4. **Table V (detection)** — mAP@0.5 and mAP@0.5:0.95 from `evaluate.py`
   (COCO eval). If you never completed the detection eval, either run it or
   drop Table V and reframe localization as a qualitative component — do not
   leave dashes in a submitted paper.
5. **Table VI (ablation)** — 5 configurations. The **BCE vs. Focal Loss** row is
   non-negotiable; without it, a reviewer will say your central methodological
   claim is unsupported.
6. **Figures 2–6** — replace each dashed placeholder box with
   `\includegraphics[width=\linewidth]{filename.png}`. You already have
   `roc_auc_curves.png`, `normalized_confusion_matrix.png`,
   `f1_accuracy_bars.png`; you still need a Grad-CAM panel and a UI screenshot.
   **Regenerate all of them at 300 DPI** (`plt.savefig(..., dpi=300,
   bbox_inches='tight')`) — screen-resolution PNGs are a common desk-reject.
7. **Detection backbone** — state which one produced the reported numbers
   (see discrepancy #1 below).
8. **Code/data availability** — add a GitHub URL if you're releasing.

---

## 4. Discrepancies between your two source documents

Your `.md` report and your `.pdf` report contradict each other in four places.
I reconciled them in the paper in a defensible way, but **you must confirm which
is true** — a reviewer who spots an internal inconsistency will distrust
everything else.

| # | `.md` report says | `.pdf` report says | How the paper handles it |
|---|---|---|---|
| 1 | Detection = Faster R-CNN + **ResNet-50**-FPN | Detection = Faster R-CNN + **MobileNetV3-Large** FPN | §IV-D presents both as two interchangeable configurations (capacity- vs. latency-optimized). **Confirm which produced your results.** |
| 2 | Classification handled by "MIMIC classifier/detector" | Classification = **DenseNet-121** | Paper uses DenseNet-121 (the `.pdf` is the more recent, research-oriented doc). |
| 3 | VinBigData = **14** classes incl. ILD, Lung Opacity, Nodule/Mass | VinBigData = **15** categories incl. Clavicle fracture, Rib fracture | Table I uses the 15-category list with a footnote naming the 14-category alternative. **Pick one.** |
| 4 | Optimizer = SGD(0.9, 5e-4) or Adam; StepLR | Optimizer = Adam(1e-3, 1e-4); ReduceLROnPlateau | Paper assigns Adam+Plateau to classification and SGD+StepLR to detection, which is the standard split and consistent with both docs. |

Also note: the official VinDr-CXR release has **22 local labels**, not 14 or 15
— so whichever subset you used, say explicitly that it is a subset.

---

## 5. The AUC problem — read this before you submit

Micro-F1 = 0.7528 with mean AUC ≈ 0.50 is **internally inconsistent**. A
reviewer will notice immediately. §VI-A of the paper explains why this happens
(498-image validation subset → several classes with ~0 positives → per-class AUC
undefined or wildly noisy; plus partial-corpus training and no calibration) and
frames it honestly as a scoping artefact rather than hiding it.

That framing is submittable, but it is **much weaker** than simply fixing it.
The cheapest fix, in order of effort:

1. **Enlarge the validation set** (a few thousand images from held-out patient
   prefixes). This alone may resolve most of it.
2. **Report per-class AUC and exclude classes with <10 positives from the mean**,
   stating the exclusion explicitly. Never average an undefined AUC as 0.5.
3. **Sanity-check the AUC call**: `roc_auc_score` must receive continuous
   **probabilities** (`sigmoid` outputs), not thresholded 0/1 predictions. If
   binary predictions were passed in, AUC collapses towards 0.5 — this is by
   far the most common cause of exactly this symptom, and it's a one-line fix.
4. **Full-corpus retrain** (Phase 1 in the roadmap table) — best result, most
   compute.

If you fix it, delete the "negative result" framing in §VI-A and update
Table III. If you can't, keep it — an honestly reported and analysed negative
result is far better received than a quietly omitted one.

---

## 6. Venue notes

| Venue | Format | Page limit | Notes |
|---|---|---|---|
| **MIDL** | Own template (not IEEEtran) | 8–10 | Highest bar; needs the full ablation + strong AUC |
| **ISBI** | IEEEtran conference | **4 pages** incl. refs | You'd need to cut ~60%; strongest fit for the current state |
| **MICCAI Workshop** | Springer LNCS | 8 | Requires LNCS class, not IEEEtran |
| **IEEE JBHI** | `\documentclass[journal]{IEEEtran}` | ~12 | Needs author bios/photos; expects full-corpus results |
| **MDPI Diagnostics** | MDPI template | flexible | Fast-track, APC applies |

The current 10-page draft is sized for a journal or a full conference track.
For ISBI, cut §II to one paragraph, merge §III/§IV, drop Tables V–VI, and keep
three figures.

---

## 7. Before you hit submit

- [ ] Anonymize if the venue is double-blind (remove authors, institution,
      acknowledgment, and any repo URL; IEEEtran: keep the `\author` block but
      replace with "Anonymous Authors").
- [ ] Every figure legible in **greyscale** and at 100% zoom.
- [ ] Every table and figure referenced in the body text ("as shown in Fig. 3").
- [ ] Abbreviations defined at first use, including in the abstract.
- [ ] Run the IEEE PDF eXpress check (embedded fonts, correct page size).
- [ ] Plagiarism/similarity check — your two source reports contain phrasing
      that may already exist elsewhere; the paper text is written fresh, but
      verify.
- [ ] Confirm you are permitted to publish results derived from MIMIC-CXR under
      your PhysioNet credentialed-access agreement (you are, but the DUA
      requires you not to redistribute the data itself).
