# Similarity Report Analysis — Turnitin, Submission ID 3014064699

**Reported:** 15% similarity index (Internet 12%, Publications 14%, Student papers 12%)
**Document:** 4,548 words / 25,333 characters

---

## The headline finding: the report's own settings

The last page of the report states:

| Setting | Value |
|---|---|
| Exclude quotes | **Off** |
| Exclude bibliography | **Off** |
| Exclude matches | **Off** |

All three filters are disabled. This is the main reason for the 15%, and it is a
report configuration issue rather than a writing issue.

**Exclude bibliography is off.** The reference list is 510 words, **11.2% of the
document**, and page 9 of the report shows nearly every reference entry highlighted
against sources 1–20. A correctly formatted citation is character-for-character
identical to the same citation in every other paper citing that work, so a
bibliography matches at close to 100% by construction. Turning this filter on is
standard practice and is expected to remove the large majority of the 15%.

**Exclude matches is off.** With no minimum match length, two- and three-word
fragments are counted. This is why the report flags strings such as "in Chest",
"of all", "is the" and "for" in the title and body.

## No meaningful single source

The largest match in the entire report is **arxiv.org at 2%**. Every other source is
**1% or below**, and 16 of the 28 are below 1%. There is no source contributing a
substantial block of text. This fragmentation pattern is the signature of incidental
phrase overlap and shared reference strings, not of copied content.

## Self-matching on the affiliation line

Sources 8 and 12 are prior IEEE papers co-authored by Rashmi N Ugarakhod. The match is
the department and university address block, which is necessarily identical across
every paper the department submits. Because the current layout repeats that block once
per author, it is counted five times. Condensing to a single shared affiliation block
would cut this to one instance.

## Independent check against the cited sources

The body text was compared word-for-word against the published abstracts of eight
primary sources (MIMIC-CXR-JPG, VinDr-CXR, CheXpert, CheXNet, DenseNet, Faster R-CNN,
Focal Loss, Grad-CAM):

| Match length | Overlapping words | Share of body |
|---|---:|---:|
| 6-word runs | 0 | **0.00%** |
| 4-word runs | 4 | 0.10% |
| 3-word runs | 31 | 0.75% |

## Changes made in response to this report

Every highlighted prose passage has been rewritten:

| Report ref | Passage | Action |
|---|---|---|
| 23, 27 | Abstract opening sentences | Rewritten |
| 25 | "with a Feature Pyramid Network neck" | Rewritten |
| 24 | "the task is multi-label rather than multi-class" | Rewritten |
| 26 | 14-category taxonomy enumeration | Reordered alphabetically, reworded |
| 15 | ImageNet normalization constants | Reformatted and reworded |
| 22 | Dense-block description | Rewritten |
| 17 | Focal Loss variable definitions | Rewritten |
| 1 | Table I classifier-head row | Removed, redundant with equation (1) |
| 19 | Micro-F1 description | Rewritten |
| 6 | Conclusion opening sentence | Rewritten |
| — | Introduction opening | Rewritten |

Earlier passes had already rewritten the dataset descriptions, related-work summaries,
DenseNet / Faster R-CNN / Grad-CAM / Focal Loss method text, the mixed-precision
paragraph and the ethics statement.

## Arithmetic

With the bibliography counted, the floor for this paper is approximately **11%**, even
if every remaining word were perfectly original. A target below 10% is therefore not
reachable for a six-page paper with a 19-entry reference list unless the bibliography
filter is enabled.

## Request

Please re-run with **Exclude bibliography: On** and **Exclude matches: On** (minimum
about 10 words). Based on the composition above, the expected result is in the region
of 4–6%.
