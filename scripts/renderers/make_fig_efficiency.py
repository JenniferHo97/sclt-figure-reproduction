"""Efficiency figure: return on trainable parameters of the sCLT-guided localized repair
vs full-layer fine-tuning. Same rank-16 LoRA (gate+up+down) and wrong-question data;
only the target-layer set differs. Trainable LoRA params per layer = 3*rank*(d_model+d_ff) except fused Phi-4;
localized targets 1--5 early layers, full-layer targets all N. Return metric = MedQA accuracy gain
over baseline per million trainable parameters (pp/M).
a gain over baseline vs trainable params (log), paired per model; b return (pp/M), log."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os
HERE=os.path.dirname(__file__)
# base, localized acc, all-layer acc, n_total_layers (Qwen3.5-0.8B has 24, not 28), d_model, d_ff, n_localized_layers
# All eight models, 182 held-out paraphrases, rank 16, 3-seed means. The localized arm edits the
# layers the binomial sign test returns, which is one to five layers depending on the model, so
# the trainable-parameter budget now varies per model rather than being fixed at three layers.
M={"Qwen-0.8B":(46.7,76.4,63.9,24,1024,3072,3),"Qwen-2B-Base":(48.9,87.4,72.0,24,2048,6144,5),
   "Qwen-2B-Inst":(53.3,84.6,71.8,24,2048,6144,3),"R1-Distill-1.5B":(26.9,29.3,35.5,28,1536,8960,2),
   "Gemma-2-2B":(38.5,65.2,37.9,26,2304,9216,2),"Gemma-3-1B":(25.8,33.5,46.2,26,1152,6912,4),
   "OLMo-2-1B":(34.6,41.6,44.7,16,2048,8192,4),"Phi-4-mini":(49.5,66.1,80.6,32,3072,8192,1)}
FUSED={"Phi-4-mini"}
RANK=16
def per_layer_M(d,ff,fused=False):
    # gate+up+down, each rank*(in+out); Phi-4 fuses gate and up into one wider matrix
    return (RANK*(2*d+3*ff) if fused else 3*RANK*(d+ff))/1e6
COL={"Qwen-0.8B":"#4477AA","Qwen-2B-Base":"#66CCEE","Qwen-2B-Inst":"#228833",
     "R1-Distill-1.5B":"#CCBB44","Gemma-3-1B":"#EE6677","Gemma-2-2B":"#AA3377",
     "OLMo-2-1B":"#BBBBBB","Phi-4-mini":"#EE9922"}
PRIN_M="D"; FULL_M="X"
CODES={"Qwen-0.8B":"Q0.8","Qwen-2B-Base":"Q2-B","Qwen-2B-Inst":"Q2-I","R1-Distill-1.5B":"R1","Gemma-2-2B":"G2","Gemma-3-1B":"G3","OLMo-2-1B":"O2","Phi-4-mini":"P4"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10.1,"axes.linewidth":0.6,
    "axes.labelsize":11.2,"xtick.labelsize":9.5,"ytick.labelsize":9.5,"legend.frameon":False})
fig,(axa,axb)=plt.subplots(1,2,figsize=(7.40,2.95),dpi=300)
def lab(a,t): a.set_title(t,loc="left",fontweight="bold",fontsize=10.5)
def clean(a):
    for s in ("top","right"): a.spines[s].set_visible(False)
# a: gain vs trainable params (M), paired
for m,(b,p,f,nl,d,ff,ll) in M.items():
    pl=per_layer_M(d,ff,m in FUSED); pm,fm=ll*pl,nl*pl; pg,fg=p-b,f-b; c=COL[m]
    axa.plot([pm,fm],[pg,fg],"-",color=c,lw=0.8,alpha=0.6,zorder=2)
    axa.scatter([pm],[pg],marker=PRIN_M,s=46,color=c,zorder=4,edgecolor="white",lw=0.5)
    axa.scatter([fm],[fg],marker=FULL_M,s=46,color=c,zorder=4)
    axa.annotate(CODES[m],(pm,pg),xytext=(-5,6),textcoords="offset points",ha="right",fontsize=8,color="#333")
axa.set_xscale("log"); axa.set_xlabel("trainable parameters (millions, log)")
axa.set_ylabel("MedQA gain over baseline (pp)"); axa.axhline(0,lw=0.5,color="#aaa")
from matplotlib.lines import Line2D
leg=[Line2D([],[],marker=PRIN_M,color="#555",ls="",ms=7,label="localized (1--5 layers)"),
     Line2D([],[],marker=FULL_M,color="#555",ls="",ms=7,label="full-layer (all)")]
fig.legend(handles=leg,loc="lower left",bbox_to_anchor=(0.08,-0.08),fontsize=9.0,ncol=2); lab(axa,"a  more gain, fewer params"); clean(axa)
# b: return = gain per million params, principled vs full, log y
mods=list(M); x=np.arange(len(mods)); w=0.36
pr=[]; fr=[]
for m in mods:
    b,p,f,nl,d,ff,ll = M[m]
    pl=per_layer_M(d,ff,m in FUSED); pr.append((p-b)/(ll*pl)); fr.append((f-b)/(nl*pl))
frp=[max(v,0.05) for v in fr]
axb.bar(x-w/2,pr,w,color=[COL[m] for m in mods])
axb.bar(x+w/2,frp,w,color=[COL[m] for m in mods],alpha=0.45,hatch="///",edgecolor="white")
for i in range(len(mods)):
    tag = "n/a" if fr[i] <= 0 else f"{pr[i]/fr[i]:.0f}x"
    axb.text(i,max(pr[i],frp[i])*1.18,tag,ha="center",fontsize=8.6,fontweight="bold",color="#333")
axb.set_yscale("log"); axb.set_ylim(0.05,700)
axb.set_xticks(x); axb.set_xticklabels([CODES[m] for m in mods],fontsize=7.4,rotation=35,ha="right")
axb.set_ylabel("return (pp / M params, log)")
axb.text(0.97,0.97,"solid: localized\nhatched: all layers",transform=axb.transAxes,
         ha="right",va="top",fontsize=7.8,color="#555"); lab(axb,"b  return on parameters"); clean(axb)
fig.tight_layout()
out=os.path.join(HERE,"F_efficiency.png"); __import__("publication_style").prepare_figure(fig, "F_efficiency"); fig.savefig(out,dpi=300,bbox_inches="tight",pad_inches=0.12,facecolor="white"); print("saved",out)
print("prin pp/M:",[round(v,1) for v in pr]); print("full pp/M:",[round(v,2) for v in fr])
print("ratios:",[("n/a" if fr[i]<=0 else round(pr[i]/fr[i],1)) for i in range(len(mods))])
