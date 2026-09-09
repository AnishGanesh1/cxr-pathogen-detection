import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

labels = ["Aortic enlarg.","Pleural thick.","Pleural effusion","Cardiomegaly",
          "Lung opacity","Nodule/mass","Consolidation","Pulm. fibrosis","Infiltration",
          "Atelectasis","Other lesion","ILD","Pneumothorax","Calcification"]
M = np.zeros((14,14), int)
M[0]  = [0,0,3,0,1,0,0,0,0,0,0,0,12,0]
M[1]  = [0,0,8,1,44,0,0,0,0,0,0,0,7,0]
M[2]  = [0,0,112,40,67,0,0,0,0,21,0,0,102,0]
M[3]  = [0,0,10,3,11,0,0,0,0,3,0,0,17,0]
M[4]  = [0,0,1,0,1,0,0,0,0,0,0,0,3,0]
M[8]  = [0,0,2,0,0,0,0,0,0,0,0,0,0,0]
M[10] = [0,0,0,0,0,0,0,0,0,0,0,0,2,0]
M[12] = [0,0,2,0,0,0,0,0,0,0,0,0,14,0]
assert M.sum()==487
print("total", M.sum(), "diagonal", np.trace(M))

SURF = "#000000"
INK, INK2 = "#ffffff", "#c3c2b7"
# sequential single hue re-stepped for the dark surface: near-surface -> light
ramp = ["#000000","#0d366b","#184f95","#256abf","#3987e5","#6da7ec","#9ec5f4","#cde2fb"]
cmap = LinearSegmentedColormap.from_list("seqblue_dark", ramp, N=256)

fig, ax = plt.subplots(figsize=(3.42, 2.62), dpi=400)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
im = ax.imshow(M, cmap=cmap, vmin=0, vmax=M.max())

ax.set_xticks(np.arange(-.5,14,1), minor=True)
ax.set_yticks(np.arange(-.5,14,1), minor=True)
ax.grid(which="minor", color=SURF, linewidth=0.8)
ax.tick_params(which="minor", length=0)

for i in range(14):
    for j in range(14):
        v = M[i,j]
        if v == 0: continue
        ax.text(j, i, str(v), ha="center", va="center", fontsize=4.4,
                color="#0b0b0b" if v > 0.55*M.max() else INK)

ax.set_xticks(range(14)); ax.set_yticks(range(14))
ax.set_xticklabels(labels, rotation=42, ha="right", fontsize=4.5, color=INK2)
ax.set_yticklabels(labels, fontsize=4.5, color=INK2)
ax.set_xlabel("Predicted top finding", fontsize=6.2, color=INK)
ax.set_ylabel("True primary finding", fontsize=6.2, color=INK)
for s in ax.spines.values(): s.set_color("#4a4a48"); s.set_linewidth(0.5)
ax.tick_params(axis="both", which="major", length=1.6, width=0.4, color="#4a4a48", pad=1.4)

cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
cb.ax.tick_params(labelsize=4.6, length=1.6, width=0.4, color="#4a4a48", colors=INK2)
cb.outline.set_edgecolor("#4a4a48"); cb.outline.set_linewidth(0.5)
cb.set_label("Studies", fontsize=5.4, color=INK2)

fig.tight_layout(pad=0.25)
fig.savefig("fig/cm_dark.png", dpi=400, facecolor=SURF, bbox_inches="tight")
print("ok")
