"""Supplementary figure: native sCLT-derived repair on PubMedQA, held-out accuracy
(base vs repaired) for the six models with an sCLT. Biomedical-literature reasoning."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os
HERE=os.path.dirname(__file__)
# Held-out base vs repaired accuracy (%), n=600 of the 1000-item PubMedQA PQA-L set, the
# remainder after 200 own-errors and 200 base-correct items are used for the repair.
# qwen2binst,gemma3,olmo2,r1distill}.json; three values had been transcribed one to two items
# high (Qwen-2B-Base, Gemma-3-1B, R1-Distill-1.5B) and are corrected here. Each model's delta
# is unchanged, so the percentages quoted in Supplementary Note 6 still hold.
P=[("Qwen-0.8B",64.0,68.2),("Qwen-2B-Base",59.0,74.8),("Qwen-2B-Inst",55.5,74.0),
   ("Gemma-3-1B",58.7,60.2),("OLMo-2-1B",47.5,70.7),("R1-Distill-1.5B",52.3,47.0)]
GREY="#cfcfcf"; ACC="#4E9E7E"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.linewidth":0.6,
    "axes.labelsize":10.5,"xtick.labelsize":9.5,"ytick.labelsize":9,"legend.frameon":False})
def clean(a):
    for s in ("top","right"): a.spines[s].set_visible(False)
fig,ax=plt.subplots(figsize=(6.2,3.2),dpi=300)
pn=[p[0] for p in P]; pb=[p[1] for p in P]; pr=[p[2] for p in P]; x=np.arange(len(P)); w=0.38
ax.bar(x-w/2,pb,w,color=GREY,label="base")
ax.bar(x+w/2,pr,w,color=ACC,label="repaired")
ax.set_xticks(x); ax.set_xticklabels(pn,fontsize=9.2,rotation=15,ha="right")
ax.set_ylim(0,90); ax.set_ylabel("PubMedQA held-out accuracy (%)")
ax.legend(loc="upper right",fontsize=9,ncol=2); clean(ax)
fig.tight_layout()
out=os.path.join(HERE,"F_pubmedqa.png"); __import__("publication_style").prepare_figure(fig, "F_pubmedqa"); fig.savefig(out,dpi=300,bbox_inches="tight",pad_inches=0.12,facecolor="white"); print("saved",out)
