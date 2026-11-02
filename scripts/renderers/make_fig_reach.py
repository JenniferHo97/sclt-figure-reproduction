"""Reachability: mean |change in CLT feature activation| per layer when a rank-16
LoRA repair edits early layers (L0,2,4) vs late layers (L20,21,22), on Qwen-0.8B
wrong questions. Early edits propagate forward and reshape the whole downstream
computation (incl. the late shortcut read-out region); late edits stay confined."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os
HERE=os.path.dirname(__file__)
# FVU-best CLT (2026-08): early edit {0,2,4}, late edit {20,21,22}; downstream mean 1.46 vs upstream 0.000
dE=[0.0,0.18,0.18,0.43,0.36,0.65,0.49,0.71,0.89,1.02,0.35,0.39,0.3,0.43,0.5,0.59,1.06,1.12,1.07,1.4,1.87,2.93,0.78,5.43]
dL=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.45,0.25,3.16]
EARLY=[0,2,4]; LATE=[20,21,22]
L=np.arange(len(dE)); w=0.42
EC="#4477AA"; LC="#cc3311"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10.6,"axes.linewidth":0.6,
    "axes.labelsize":11.9,"xtick.labelsize":9.2,"ytick.labelsize":10.0,"legend.frameon":False})
fig,ax=plt.subplots(figsize=(6.20,2.93),dpi=300)
ax.bar(L-w/2,dE,w,color=EC,label="early-layer edit (L0,2,4)")
ax.bar(L+w/2,dL,w,color=LC,label="late-layer edit (L20,21,22)")

ax.axvspan(11.5,23.5,color="#f2f2f2",zorder=0)
ax.text(17.5,5.3,"late shortcut\nread-out region",ha="center",fontsize=8.8,color="#777")
ax.set_xlabel("transformer layer"); ax.set_ylabel("mean $|\\Delta$ feature activation$|$")
ax.set_xticks(L); ax.set_xlim(-0.6,23.6); ax.set_ylim(-0.7,6.2)
ax.legend(loc="upper left",fontsize=10.0)
for s in ("top","right"): ax.spines[s].set_visible(False)
fig.tight_layout()
out=os.path.join(HERE,"F_reach.png"); __import__("publication_style").prepare_figure(fig, "F_reach"); fig.savefig(out,dpi=300,bbox_inches="tight",pad_inches=0.12,facecolor="white"); print("saved",out)
