"""Nature-style multi-model composite for Results 2.1, data-driven from
clt_results.json. Panels a-b compare our Gemma-2-2B Local CLT to the standard
full-width CLT; panels c-k give per-model / per-layer detail (no standard CLT).
Laid out 4 rows x 3 cols at a print-oriented figsize so fonts stay legible when
the figure is placed at the single-column text width (~5.15 in)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter, LogLocator
import numpy as np, json, os, statistics as st

HERE = os.path.dirname(__file__)
DATA=os.path.abspath(os.path.join(HERE,"..","..","data"))
data = json.load(open(os.path.join(DATA, "clt_results.json"), encoding="utf-8"))
order = ["gemma2-2b","gemma3-1b","qwen35-0.8b","qwen35-2b-base","qwen35-2b-inst",
         "r1-distill-1.5b","olmo2-1b","phi4-mini"]
_HF = data.get("_hf_gemma2_ref")
data = {k: v for k, v in data.items() if not k.startswith("_")}
models = [m for m in order if m in data] + [m for m in data if m not in order]
DISP = {"qwen35-0.8b":"Qwen-0.8B","qwen35-2b-base":"Qwen-2B-Base","qwen35-2b-inst":"Qwen-2B-Inst",
        "gemma2-2b":"Gemma-2-2B","gemma3-1b":"Gemma-3-1B","olmo2-1b":"OLMo-2-1B",
        "r1-distill-1.5b":"R1-Distill-1.5B","phi4-mini":"Phi-4-mini"}
CODE={"gemma2-2b":"G2","gemma3-1b":"G3","qwen35-0.8b":"Q0.8","qwen35-2b-base":"Q2-B","qwen35-2b-inst":"Q2-I","r1-distill-1.5b":"R1","olmo2-1b":"O2","phi4-mini":"P4"}
def nm(m): return CODE.get(m, m)
# canonical soft, family-grouped per-model palette (Qwen=blues, Gemma=greens, others distinct)
CMODEL = {"gemma2-2b":"#4E9E7E","gemma3-1b":"#9FCDA9","qwen35-0.8b":"#4E86B0",
          "qwen35-2b-base":"#A6C6DE","qwen35-2b-inst":"#2F5E86","r1-distill-1.5b":"#A56FA0",
          "olmo2-1b":"#AFA79D","phi4-mini":"#E0A15F"}
COL = {m: CMODEL.get(m, "#B4B4B4") for m in models}
GREY = "#d5d5d5"; HFC = "#7FB0AD"; OURS = "#4E9E7E"
# Explicit model styles apply to the depth curves and their shared key only.
# Model identity in the one-value bar panels is already given by the row label.
MODEL_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
MODEL_LINES = ("-", "--", "-.", ":", "-", "--", "-.", ":")
MODEL_STYLE = {m: dict(marker=MODEL_MARKERS[i], linestyle=MODEL_LINES[i], markersize=3.0)
               for i,m in enumerate(order)}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.6,
    "axes.labelsize": 9.5, "xtick.labelsize": 8, "ytick.labelsize": 7.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "legend.frameon": False, "lines.solid_capstyle": "round",
})
VLAB = 6.8   # on-bar value annotations
def frac(n): return np.linspace(0, 1, n)
def clean(ax):
    for s in ("top","right"): ax.spines[s].set_visible(False)
def lab(ax, L): ax.set_title(L, loc="left", fontweight="bold", fontsize=11)

fig = plt.figure(figsize=(7.6, 8.8), dpi=300)
gs = fig.add_gridspec(4, 3, left=0.11, right=0.985, top=0.925, bottom=0.05,
                      hspace=0.62, wspace=0.62)
axa,axb,axc = (fig.add_subplot(gs[0,i]) for i in range(3))
axd,axe,axf = (fig.add_subplot(gs[1,i]) for i in range(3))
axg,axh,axi = (fig.add_subplot(gs[2,i]) for i in range(3))
axj = fig.add_subplot(gs[3,0]); axk = fig.add_subplot(gs[3,1])
axleg = fig.add_subplot(gs[3,2]); axleg.axis("off")

# ---- a: Gemma-2-2B Local CLT vs standard CLT, all metrics (same-direction grouped) ----
g2 = data["gemma2-2b"]
mets   = ["FVU", "dead\n%", "cosine"]
ours_v = [g2["mean_fvu"], g2["mean_dead_frac"]*100, g2["mean_cos"]]
std_v  = [_HF["fvu"], _HF["mean_dead_frac"]*100, _HF["mean_cos"]]
xa = np.arange(len(mets)); wa = 0.38
axa.bar(xa-wa/2, ours_v, wa, color=OURS, edgecolor="white", linewidth=0.4, label="sCLT (ours)")
axa.bar(xa+wa/2, std_v,  wa, color=HFC, edgecolor="#637a78", linewidth=0.4, hatch="///", label="full CLT")
axa.set_yscale("log"); axa.set_ylim(0.05, 400)
def vfmt(v): return f"{v:.2f}" if v < 10 else f"{v:.0f}"
for xx_,v in zip(xa-wa/2, ours_v): axa.text(xx_, v*1.22, vfmt(v), ha="center", fontsize=VLAB, color=OURS)
for metric_index,(xx_,v,o) in enumerate(zip(xa+wa/2, std_v, ours_v)):
    # two bars of nearly equal height would put both labels side by side and run them together
    lift = 2.4 if abs(np.log10(v/o)) < 0.15 else 1.22
    label=axa.text(xx_, v*lift, vfmt(v), ha="center", fontsize=VLAB, color=HFC)
    if metric_index == 2:
        # Separate 0.83 from the adjacent 0.92 without changing either bar.
        # At the manuscript width this 6.6-source-pt offset is about 4.5 pt.
        label.set_transform(label.get_transform() + matplotlib.transforms.ScaledTranslation(
            0, 6.6/72, fig.dpi_scale_trans))
axa.set_xticks(xa); axa.set_xticklabels(mets, fontsize=7.6)
lab(axa,"a"); clean(axa)

# ---- b: size vs the official Gemma-2-2B CLT (same model as panel a) ----
# Reference = mntss/clt-gemma-2-2b-426k on Hugging Face: 26 layers x 16,384 features,
# d_model 2304, encoder 26 blocks + cross-layer decoder 351 (=26*27/2) blocks of 2304x16384
# -> 425,984 features, 1.42e10 parameters (read off the safetensors shapes, 2026-09-30).
# Its training-token count is not published, so tokens are not compared.
HF_FEAT = 26 * 16384
HF_PARAMS = (26 + 26 * 27 // 2) * 2304 * 16384
lp = g2["params_M"] * 1e6
lf = sum(g2["features_per_layer"])
cats = ["parameters", "features"]
locv = [lp, lf]; stdv = [HF_PARAMS, HF_FEAT]
xb = np.arange(len(cats)); wb = 0.37
axb.bar(xb-wb/2, locv, wb, color=OURS, edgecolor="white", linewidth=0.4, label="sCLT (ours)")
axb.bar(xb+wb/2, stdv, wb, color=HFC, edgecolor="#637a78", linewidth=0.5, hatch="///", label="full CLT")
axb.set_yscale("log"); axb.set_xticks(xb); axb.set_xticklabels(cats, fontsize=8); axb.set_ylim(1e4, 3e11)
for i in range(len(cats)):
    axb.text(xb[i], stdv[i]*2.3, f"{stdv[i]/locv[i]:.1f}$\\times$", ha="center", fontsize=8.5, color="#777")
lab(axb,"b"); clean(axb)
fig.legend(handles=[Patch(facecolor=OURS,edgecolor="white",label="sCLT (ours)"),
                    Patch(facecolor=HFC,edgecolor="#637a78",hatch="///",label="full CLT")],
           loc="lower left",bbox_to_anchor=(0.105,0.945),ncol=2,fontsize=8.3,
           frameon=False,handlelength=1.3,columnspacing=1.5)

# ---- c: per-model mean FVU + L0 (no standard CLT) ----
y = np.arange(len(models))
fvus = [data[m].get("mean_fvu") for m in models]; l0s = [data[m].get("mean_l0") for m in models]
errs = [data[m].get("fvu_std") or 0 for m in models]
axc.barh(y, fvus, 0.7, xerr=errs, color=[COL[m] for m in models], edgecolor="white", linewidth=0.5,
         error_kw=dict(lw=0.8, capsize=2, ecolor="#555"))
for i,v in enumerate(fvus):
    if v is not None:
        axc.text(v+errs[i]+0.02, i, f"{v:.2f}", va="center", fontsize=VLAB, color="#333")
axc.set_yticks(y); axc.set_yticklabels([nm(m) for m in models]); axc.invert_yaxis()
axc.set_xlabel("mean FVU"); axc.set_xlim(0, 1.0)
lab(axc,"c"); clean(axc)

# ---- line panels (no standard CLT) ----
def lines(ax, key, ylab, log=False, sci=False, unit=False):
    for m in models:
        v = data[m].get(key)
        if v: ax.plot(frac(len(v)), v, lw=1.3, color=COL[m], alpha=0.9,
                      markevery=max(1,len(v)//5), **MODEL_STYLE[m])
    ax.set_xlabel("relative depth"); ax.set_ylabel(ylab); ax.set_xlim(0,1)
    if unit: ax.set_ylim(0,1)
    if log: ax.set_yscale("log")
    if sci: ax.ticklabel_format(axis="y", style="sci", scilimits=(3,3))
    clean(ax)

lines(axd, "fvu_per_layer", "FVU", unit=True); lab(axd,"d")
lines(axe, "l0_per_layer", "active / token", log=True); lab(axe,"e")
lines(axf, "dims_per_layer", "dims / layer", sci=True); lab(axf,"f")
lines(axg, "features_per_layer", "CLT features / layer", sci=True); lab(axg,"g")

# ---- h: per-model totals; legend order matches bar order; x ticks at decades ----
sd  = [sum(data[m].get("dims_per_layer") or [0]) for m in models]
sfe = [sum(data[m].get("features_per_layer") or [0]) for m in models]
axh.barh(y-0.19, sfe, 0.34, color=[COL[m] for m in models], edgecolor="#666666", linewidth=0.5, hatch="///")
axh.barh(y+0.19, sd, 0.34, color=[COL[m] for m in models], edgecolor="#666666", linewidth=0.4)
axh.set_xscale("log"); axh.set_yticks(y); axh.set_yticklabels([nm(m) for m in models])
axh.invert_yaxis(); axh.set_xlabel("count (log)")
axh.xaxis.set_major_locator(LogLocator(base=10)); axh.xaxis.set_minor_formatter(NullFormatter())
axh.legend(handles=[Patch(facecolor="#b9b9b9",edgecolor="#666666",hatch="///",label="features"),
                    Patch(facecolor="#b9b9b9",edgecolor="#666666",label="dims")],
           fontsize=8.0, loc="lower right", bbox_to_anchor=(1.0,1.015), ncol=2,
           frameon=False, handlelength=0.9, handletextpad=0.25,
           columnspacing=0.45, borderpad=0.1)
lab(axh,"h"); clean(axh)

# ---- i: cosine by layer ----
if any(data[m].get("cos_per_layer") for m in models):
    lines(axi, "cos_per_layer", "cosine", unit=True)
lab(axi,"i")

# ---- j: dead-feature fraction + density per model (no standard CLT) ----
h2 = 0.38
dead=[(data[m].get("mean_dead_frac") or 0)*100 for m in models]
dens=[(data[m].get("mean_density") or 0)*100 for m in models]
icol=[COL[m] for m in models]
axj.barh(y+h2/2,dead,h2,color=icol,edgecolor="white",linewidth=0.3)

for k in range(len(y)):
    axj.text(dead[k]+1.2,y[k]+h2/2,f"{dead[k]:.0f}",va="center",fontsize=VLAB,color="#444")
    # density values sit in a right-hand column: beside their thin bars they collided with the dead-% labels
    axj.text(106,y[k]+h2/2,f"{dens[k]:.1f}",va="center",ha="right",fontsize=VLAB,color="#777")
axj.text(106,-0.95,"density %",ha="right",va="bottom",fontsize=VLAB-0.4,color="#777")
axj.set_yticks(y); axj.set_yticklabels([nm(m) for m in models]); axj.invert_yaxis()
axj.set_xlabel("dead features (%)"); axj.set_xlim(0, 108)
lab(axj,"j"); clean(axj)

# ---- k: cosine per model (no standard CLT) ----
cosm=[data[m].get("mean_cos") or 0 for m in models]
axk.barh(y,cosm,0.62,color=[COL[m] for m in models],edgecolor="white",linewidth=0.4)
for k in range(len(y)):
    if cosm[k]: axk.text(cosm[k]+0.02,y[k],f"{cosm[k]:.2f}",va="center",fontsize=VLAB,color="#444")
axk.set_yticks(y); axk.set_yticklabels([nm(m) for m in models]); axk.invert_yaxis()
axk.set_xlim(0,1.2); axk.set_xlabel("cosine")
lab(axk,"k"); clean(axk)

# ---- shared model legend ----
handles = [Line2D([0],[0], color=COL[m], lw=2.0, **MODEL_STYLE[m]) for m in models]
axleg.legend(handles, [CODE[m]+": "+DISP[m] for m in models], loc="center", fontsize=8, handlelength=1.6,
             title="model code", title_fontsize=9)

for a in (axd,axe,axf,axg,axi): a.set_xticks([0,0.5,1])
axc.set_xticks([0,0.5,1])
out = os.path.join(HERE, "F_clt.png")
def reflow(figure):
    # Legend bands and row spacing are set after type-size convergence.
    source_font=max(t.get_fontsize() for t in axleg.get_legend().get_texts())
    gs.update(left=0.11,right=0.985,top=0.925,bottom=0.05,
              hspace=max(0.70,0.70*source_font/10.6),wspace=0.68)

__import__("publication_style").prepare_figure(fig, "F_clt", layout=reflow)
for _ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(HERE,"F_clt."+_ext),dpi=300,bbox_inches="tight",pad_inches=0.12,facecolor="white")
import shutil as _sh
for _ext in ("png", "pdf", "svg"):
    _sh.copyfile(os.path.join(HERE,"F_clt."+_ext),os.path.join(os.path.dirname(HERE),"sn-article-template","F_clt."+_ext))
print(f"saved {out} | {len(models)} models")
