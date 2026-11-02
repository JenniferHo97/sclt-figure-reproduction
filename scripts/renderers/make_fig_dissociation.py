"""Dissociation proof (Qwen-0.8B): the layers most causally relevant to the decision
at inference (read-out, late) are NOT the layers where fine-tuning repairs it (early).
E(L): MedQA accuracy of a single-layer rank-16 LoRA at layer L (base 42.5%).
R(L): per-layer read-out relevance = sum of |Δp_gold| of that layer's key thief/hero
      features from the per-question causal search."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os, shutil
HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.path.abspath(os.path.join(HERE,"..","..","data"))
# E = MedQA accuracy of a single-layer rank-16 LoRA at that layer of Qwen-0.8B, scored on its
# 200-question audit; R = per-layer Sigma|dp_gold| of that layer's key thief and hero features.
# E re-run 2026-09-28 with alpha=beta=0, the objective every arm of Fig. 5 uses; the previous
# values came from a 2026-08 run under alpha=0.02, beta=0.05 that no longer exists on disk.
E=[0.785,0.765,0.795,0.750,0.700,0.750,0.755,0.685,0.665,0.640,0.600,0.490,0.460,0.430,0.385,0.390,0.410,0.390,0.390,0.385,0.385,0.360,0.305,0.405]
R=[0.35,0.0,0.542,0.0,0.194,0.017,0.0,0.0,0.0,0.0,0.0,0.0,0.112,0.483,0.0,0.03,0.465,0.009,0.074,0.084,0.23,0.008,0.471,0.51]
BASE=0.425; L=np.arange(24); N=24
E=np.array(E)*100; R=np.array(R)
r=np.corrcoef(E,R)[0,1]
from scipy.stats import spearmanr
rho=spearmanr(E,R).correlation if True else 0
GREEN="#228833"; RED="#cc3311"
# layout (2026-09-11): the PNG is sized to print at ~1:1 at \textwidth (16.6 cm), so the three
# text sizes below are the printed sizes: panel titles, axis labels, everything else.
FT,FL,FS=10.0,9.0,8.0
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":FS,"axes.linewidth":0.6,
    "axes.labelsize":FL,"xtick.labelsize":FS,"ytick.labelsize":FS,"legend.frameon":False,"legend.fontsize":FS})
fig,(axa,axb)=plt.subplots(1,2,figsize=(6.45,2.55),dpi=300,gridspec_kw={"width_ratios":[1.32,1]})
def lab(a,t): a.set_title(t,loc="left",fontweight="bold",fontsize=FT)
def clean(a,sides=("top","right")):
    for s in sides: a.spines[s].set_visible(False)
def num(v): return ("%.2f"%v).replace("-","−")
# a: E(L) and R(L) vs depth
axa.axvspan(-0.5,N/2-0.5,color="#eef7f0",zorder=0)
axa.text(N/4-0.5,98.5,"early half",ha="center",va="top",fontsize=FS,color=GREEN)
axa.axhline(BASE*100,ls=":",lw=0.9,color="#888")
axa.text(N/2-0.55,BASE*100+1.0,"base 42.5%",ha="right",va="bottom",fontsize=FS,color="#555")
l1=axa.plot(L,E,"-o",color=GREEN,lw=1.6,ms=4,mfc="white",mec=GREEN,label="E(L)")[0]
axa.set_xlabel("layer L (depth)"); axa.set_ylabel("edit efficacy E(L) (%)",color=GREEN); axa.tick_params(axis="y",labelcolor=GREEN)
axa.set_ylim(30,100); axa.set_yticks([40,60,80,100]); axa.set_xlim(-0.5,23.5); axa.set_xticks(range(0,24,4))
ax2=axa.twinx()
l2=ax2.plot(L,R,"-s",color=RED,lw=1.4,ms=3.5,mfc="white",mec=RED,label="R(L)")[0]
ax2.set_ylabel("read-out relevance R(L)",color=RED); ax2.tick_params(axis="y",labelcolor=RED)
ax2.set_ylim(-0.04,0.95); ax2.set_yticks([0,0.2,0.4,0.6,0.8])
clean(axa); clean(ax2,("top","left","bottom"))
ax2.legend(handles=[l1,l2],loc="upper right",ncol=2,handlelength=1.6,handletextpad=0.4,columnspacing=0.8,borderaxespad=0.3)
lab(axa,"a  edit early, read-out late")
# b: scatter E vs R
sc=axb.scatter(R,E,s=20,c=L,cmap="viridis",edgecolors="white",linewidths=0.5,zorder=3)
m,b=np.polyfit(R,E,1); xs=np.linspace(0,R.max(),50)
axb.plot(xs,m*xs+b,"--",color="#555",lw=1,zorder=2)
cb=fig.colorbar(sc,ax=axb,fraction=0.05,pad=0.04,ticks=range(0,24,4))
cb.set_label("layer depth",fontsize=FL); cb.ax.tick_params(labelsize=FS); cb.outline.set_linewidth(0.6)
axb.set_xlabel("read-out relevance R(L)"); axb.set_ylabel("edit efficacy E(L) (%)")
axb.set_xlim(-0.03,0.58); axb.set_ylim(32,84)
axb.text(0.57,59.0,"Pearson $r$ = %s\nSpearman $\\rho$ = %s"%(num(r),num(rho)),ha="right",va="center",fontsize=FS,zorder=4)
lab(axb,"b  E(L) vs. R(L): decoupled"); clean(axb)
fig.tight_layout(w_pad=2.4)
out=os.path.join(HERE,"F_dissociation.png"); __import__("publication_style").prepare_figure(fig, "F_dissociation"); fig.savefig(out,dpi=300,bbox_inches="tight",pad_inches=0.12,facecolor="white")
print("saved",out,"| Pearson",round(r,3),"Spearman",round(rho,3))
TPL=os.path.join(HERE,"..","sn-article-template","F_dissociation.png")
if os.path.isdir(os.path.dirname(TPL)): shutil.copyfile(out,TPL); print("copied",os.path.normpath(TPL))
