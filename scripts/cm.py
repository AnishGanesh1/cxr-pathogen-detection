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
assert M.sum()==487, M.sum()
print("total", M.sum(), "diagonal", np.trace(M))

# single-hue sequential ramp (light -> dark), surface white at zero
ramp = ["#ffffff","#cde2fb","#9ec5f4","#6da7ec","#3987e5","#256abf","#184f95","#0d366b"]
cmap = LinearSegmentedColormap.from_list("seqblue", ramp, N=256)

INK, INK2 = "#0b0b0b", "#52514e"
fig, ax = plt.subplots(figsize=(3.42, 2.62), dpi=400)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
im = ax.imshow(M, cmap=cmap, vmin=0, vmax=M.max())

# 2px surface gap between cells
ax.set_xticks(np.arange(-.5,14,1), minor=True)
ax.set_yticks(np.arange(-.5,14,1), minor=True)
ax.grid(which="minor", color="#d0cfcb", linewidth=0.35)
ax.tick_params(which="minor", length=0)

for i in range(14):
    for j in range(14):
        v = M[i,j]
        if v == 0: continue                       # blank == zero; no number on every cell
        ax.text(j, i, str(v), ha="center", va="center", fontsize=4.4,
                color="white" if v > 0.55*M.max() else INK)

ax.set_xticks(range(14)); ax.set_yticks(range(14))
ax.set_xticklabels(labels, rotation=42, ha="right", fontsize=4.5, color=INK2)
ax.set_yticklabels(labels, fontsize=4.5, color=INK2)
ax.set_xlabel("Predicted top finding", fontsize=6.2, color=INK)
ax.set_ylabel("True primary finding", fontsize=6.2, color=INK)
for s in ax.spines.values(): s.set_color("#b4b3af"); s.set_linewidth(0.5)
ax.tick_params(axis="both", which="major", length=1.6, width=0.4, color="#d5d4d0", pad=1.4)

cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
cb.ax.tick_params(labelsize=4.6, length=1.6, width=0.4, color="#d5d4d0", colors=INK2)
cb.outline.set_edgecolor("#d5d4d0"); cb.outline.set_linewidth(0.5)
cb.set_label("Studies", fontsize=5.4, color=INK2)

fig.tight_layout(pad=0.25)
fig.savefig("fig/cm_white.png", dpi=400, facecolor="white", bbox_inches="tight")
print("ok")
