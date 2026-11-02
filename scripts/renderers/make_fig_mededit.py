"""MedEditBench: base vs CLT-repaired across the three knowledge-editing probes
(reliability, generalisation, robustness), logit-based scoring. gemma3 repaired
is excluded (degenerate output on this prompt format). 1x3, shared y-axis so only
the leftmost panel carries model names (legible at A4 single-column width)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, json, os
HERE=os.path.dirname(__file__)
DATA=os.path.abspath(os.path.join(HERE,"..","..","data"))
S=json.load(open(os.path.join(DATA,"mededit_summary.json")))
DISP={"q2b":"Qwen-2B-Base","q2i":"Qwen-2B-Inst","qwen08":"Qwen-0.8B","gemma2":"Gemma-2-2B",
      "r1":"R1-Distill-1.5B","olmo":"OLMo-2-1B","phi4":"Phi-4-mini"}
models=["q2b","q2i","qwen08","gemma2","r1","olmo","phi4"]   # gemma3 excluded (rep degenerate)
probes=[("a  reliability","reliability"),("b  generalisation","generalization"),("c  robustness","robustness")]
BASE="#bbbbbb"; REP="#228833"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,"axes.linewidth":0.7,
    "axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8.2,"legend.frameon":False})
fig,axs=plt.subplots(1,3,figsize=(7.2,2.7),dpi=300,sharey=True)
y=np.arange(len(models)); h=0.38
for ax,(title,key) in zip(axs,probes):
    b=[S[f"{m}_base"][key] for m in models]; r=[S[f"{m}_rep"][key] for m in models]
    ax.barh(y+h/2,b,h,color=BASE,edgecolor="white",linewidth=0.4,label="base")
    ax.barh(y-h/2,r,h,color=REP,edgecolor="white",linewidth=0.4,label="repaired")
    for i in range(len(models)):
        ax.text(b[i]+0.008,y[i]+h/2,f"{b[i]:.2f}",va="center",fontsize=6.2,color="#666")
        ax.text(r[i]+0.008,y[i]-h/2,f"{r[i]:.2f}",va="center",fontsize=6.2,color=REP)
    ax.set_xlim(0,0.78); ax.set_xlabel(key.replace("ization","isation")+" ($\\uparrow$)")
    ax.set_title(title,loc="left",fontweight="bold",fontsize=10.5)
    for s in ("top","right"): ax.spines[s].set_visible(False)
axs[0].set_yticks(y); axs[0].set_yticklabels([DISP[m] for m in models]); axs[0].invert_yaxis()
fig.legend(*axs[2].get_legend_handles_labels(),fontsize=8,loc="lower center",bbox_to_anchor=(0.56,-0.08),ncol=2)
fig.tight_layout(w_pad=1.0)
out=os.path.join(HERE,"F_mededit.png"); __import__("publication_style").prepare_figure(fig, "F_mededit"); fig.savefig(out,dpi=300,bbox_inches="tight",pad_inches=0.12,facecolor="white")
print("saved",out)
