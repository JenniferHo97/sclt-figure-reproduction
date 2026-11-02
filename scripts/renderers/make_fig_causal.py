"""Figure 3 (Results 2.2): per-question causal search, 4-panel composite, 2 rows x 2 cols
(designed for the 5.15in single-column text width -> legible fonts).
a thief heatmap; b hero heatmap; c thief early/mid/late; d hero early/mid/late.
Repair-target layers and audit counts moved to Methods / Supplementary."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np, json, os
SOFTSEQ=LinearSegmentedColormap.from_list("softseq",["#FBF3E8","#EFC79A","#E0A15F","#D98E73","#9B6A9E","#5E3F6B"])
HERE=os.path.dirname(__file__)
DATA=os.path.abspath(os.path.join(HERE,"..","..","data"))
data=json.load(open(os.path.join(DATA,"causal_all.json"),encoding="utf-8"))
DISP={"qwen35-0.8b":"Qwen-0.8","qwen35-2b-base":"Qwen-2-B","qwen35-2b-inst":"Qwen-2-I",
      "gemma2-2b":"Gemma-2","gemma3-1b":"Gemma-3","olmo2-1b":"OLMo-2","r1-distill-1.5b":"R1-1.5",
      "phi4-mini":"Phi-4"}
order=["gemma2-2b","gemma3-1b","qwen35-0.8b","qwen35-2b-base","qwen35-2b-inst","r1-distill-1.5b","olmo2-1b","phi4-mini"]
models=[m for m in order if m in data]
NB=10; EARLY="#4E86B0"; MID="#D6C97A"; LATE="#D98E73"
DEPTH_STYLE={"early":dict(color=EARLY,hatch=""),
             "mid":dict(color=MID,hatch="///"),
             "late":dict(color=LATE,hatch="\\\\")}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":0.6,"axes.labelsize":9.5,
    "xtick.labelsize":8,"ytick.labelsize":8.5,"legend.frameon":False})
TL=11.5; YT=8.5
def heat(which):
    M=np.zeros((len(models),NB))
    for i,m in enumerate(models):
        nl=data[m]["n_layers"]
        for L,c in data[m][which].items(): M[i,min(int(float(L)/max(nl-1,1)*NB),NB-1)]+=c
        if M[i].sum(): M[i]/=M[i].sum()/100.0
    return M
def split3(m,which):
    nl=data[m]["n_layers"]; d=data[m][which]; tot=sum(d.values()) or 1
    e=sum(v for k,v in d.items() if int(float(k))<nl/3)
    l=sum(v for k,v in d.items() if int(float(k))>=2*nl/3)
    return 100*e/tot,100*(tot-e-l)/tot,100*l/tot
fig=plt.figure(figsize=(7.2,6.5),dpi=300)
gs=fig.add_gridspec(3,2,height_ratios=[1,1,0.13],left=0.15,right=0.98,
                    top=0.95,bottom=0.04,hspace=0.55,wspace=0.60)
axa=fig.add_subplot(gs[0,0]); axb=fig.add_subplot(gs[0,1])
axc=fig.add_subplot(gs[1,0]); axd=fig.add_subplot(gs[1,1])
def lab(a,t): a.set_title(t,loc="left",fontweight="bold",fontsize=TL)
def clean(a):
    for s in ("top","right"): a.spines[s].set_visible(False)
ylabels=[DISP[m] for m in models]; y=np.arange(len(models))
ext=[0,1,len(models)-0.5,-0.5]
for axx,which,t in [(axa,"thief","a  thief features"),(axb,"hero","b  hero features")]:
    M=heat(which); im=axx.imshow(M,aspect="auto",cmap="Greys",extent=ext,vmin=0,vmax=max(40,M.max()))
    axx.axvline(0.5,ls="--",lw=0.9,color="#6a6a6a")
    axx.set_yticks(y); axx.set_yticklabels(ylabels,fontsize=YT); axx.set_xticks([0,0.5,1]); axx.set_xlabel("relative depth"); lab(axx,t)
    cb=fig.colorbar(im,ax=axx,fraction=0.046,pad=0.03); cb.ax.tick_params(labelsize=7)
for axx,which,t in [(axc,"thief","c  thief early/mid/late"),(axd,"hero","d  hero early/mid/late")]:
    E=[];Md=[];La=[]
    for m in models: a3=split3(m,which); E.append(a3[0]); Md.append(a3[1]); La.append(a3[2])
    axx.barh(y,E,label="early",edgecolor="#666666",linewidth=0.4,**DEPTH_STYLE["early"])
    axx.barh(y,Md,left=E,label="mid",edgecolor="#666666",linewidth=0.4,**DEPTH_STYLE["mid"])
    axx.barh(y,La,left=np.array(E)+np.array(Md),label="late",edgecolor="#666666",linewidth=0.4,**DEPTH_STYLE["late"])
    axx.set_yticks(y); axx.set_yticklabels(ylabels,fontsize=YT); axx.invert_yaxis(); axx.set_xlim(0,100)
    axx.set_xlabel("% of features"); lab(axx,t); clean(axx)
axleg=fig.add_subplot(gs[2,:]); axleg.axis("off")
axleg.legend(handles=[Patch(facecolor=v["color"],hatch=v["hatch"],edgecolor="#666666",linewidth=0.4,label=k)
                      for k,v in DEPTH_STYLE.items()],
             loc="center",ncol=3,fontsize=8.5,frameon=False,handlelength=1.5,columnspacing=2.0)
def reflow(figure):
    legend_font=max(t.get_fontsize() for t in axleg.get_legend().get_texts())
    gs.set_height_ratios([1,1,max(0.13,0.13*legend_font/10.5)])
    gs.update(left=0.15,right=0.98,top=0.95,bottom=0.04,hspace=0.55,wspace=0.60)

out=os.path.join(HERE,"F_causal.png")
__import__("publication_style").prepare_figure(fig, "F_causal", layout=reflow)
for _ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(HERE,"F_causal."+_ext),dpi=300,bbox_inches="tight",pad_inches=0.12,facecolor="white")
print("saved",out)
import shutil as _sh
for _ext in ("png", "pdf", "svg"):
    _sh.copyfile(os.path.join(HERE,"F_causal."+_ext),os.path.join(os.path.dirname(HERE),"sn-article-template","F_causal."+_ext))
print("synced -> sn-article-template/F_causal.{png,pdf,svg}")
