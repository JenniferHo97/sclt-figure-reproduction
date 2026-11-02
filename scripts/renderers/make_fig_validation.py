"""Supplementary Note 2 figure: causal validation of the shortcut features, all eight models.
a hero dose-response; b flips at each dose; c enrichment, Mantel-Haenszel and replication counts;
d per-feature log-odds with 95% CI; e held-out transfer margin gain against random features;
f held-out flips against random. Data-driven from validation_results.json, written by
aggregate_validation.py on the server; every model uses the transcoder its held-out repair reads."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np, json, os

HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.path.abspath(os.path.join(HERE,"..","..","data"))
data = json.load(open(os.path.join(DATA, "validation_results.json"), encoding="utf-8"))
order = ["gemma2-2b", "gemma3-1b", "qwen35-0.8b", "qwen35-2b-base", "qwen35-2b-inst",
         "r1-distill-1.5b", "olmo2-1b", "phi4-mini"]
models = [m for m in order if m in data]
DISP = {"qwen35-0.8b": "Qwen-0.8", "qwen35-2b-base": "Qwen-2-B", "qwen35-2b-inst": "Qwen-2-I",
        "gemma2-2b": "Gemma-2", "gemma3-1b": "Gemma-3", "olmo2-1b": "OLMo-2",
        "r1-distill-1.5b": "R1-1.5", "phi4-mini": "Phi-4"}
# the model palette of Fig. 3, so a model keeps one colour throughout the paper
COL = {"gemma2-2b": "#4E9E7E", "gemma3-1b": "#9FCDA9", "qwen35-0.8b": "#4E86B0", "qwen35-2b-base": "#A6C6DE",
       "qwen35-2b-inst": "#2F5E86", "r1-distill-1.5b": "#A56FA0", "olmo2-1b": "#AFA79D", "phi4-mini": "#E0A15F"}
RAND = "#BDBDBD"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.6,
                     "axes.labelsize": 9.5, "xtick.labelsize": 8, "ytick.labelsize": 8.5,
                     "legend.frameon": False})
TL = 11.5; VAL = 6.8


def lab(a, t): a.set_title(t, loc="left", fontweight="bold", fontsize=TL)


def clean(a):
    for s in ("top", "right"): a.spines[s].set_visible(False)


# Legends of the bar panels go inside, at the upper right where the bars are short. Above the axes
# they overprinted the panel title however far right they were anchored, because they are wider than
# the space left of the title. ncol is kept in the signature but forced to one column here.
def toplegend(ax, handles, ncol):
    ax.legend(handles=handles, fontsize=6.2, loc="upper center", ncol=3,
              handlelength=1.2, labelspacing=0.35, borderaxespad=0.1, bbox_to_anchor=(0.5,0.99))


fig = plt.figure(figsize=(7.4, 8.6), dpi=300)
gs = fig.add_gridspec(3, 2, left=0.16, right=0.97, top=0.955, bottom=0.17, hspace=0.58, wspace=0.62)
axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
axc, axd = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
axe, axf = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
y = np.arange(len(models)); names = [DISP[m] for m in models]

# ---- a, b: hero dose-response. Absolute gains on a log axis: normalised curves collapsed onto
# one line and hid that the models differ twenty-fold in scale.
nmono = 0
for m in models:
    dz = data[m].get("dose")
    if not dz: continue
    nmono += bool(dz["monotonic"])
    axa.plot(dz["scales"], dz["margin"], "-o", ms=3.2, lw=1.2, color=COL[m])
    axb.plot(dz["scales"], dz["flip_pct"], "-o", ms=3.2, lw=1.2, color=COL[m])
axa.set_yscale("log"); axa.set_xticks([1.5, 2.0, 3.0]); axa.set_xlim(1.35, 3.15)
axa.set_xlabel("hero amplification $s$"); axa.set_ylabel("mean margin gain")
axa.text(0.03, 0.97, "monotonic on %d/%d models" % (nmono, len(models)), transform=axa.transAxes,
         ha="left", va="top", fontsize=VAL, color="#555")
lab(axa, "a  hero dose-response"); clean(axa)
axb.set_xticks([1.5, 2.0, 3.0]); axb.set_xlim(1.35, 3.15); axb.set_ylim(0, 105)
axb.set_xlabel("hero amplification $s$"); axb.set_ylabel("errors flipped to correct (%)")
lab(axb, "b  flips by dose"); clean(axb)

# ---- c: enrichment / MH / replication counts ----
hc = 0.26
sig = [data[m].get("stats_letterconf", {}).get("n_sig", 0) for m in models]
mh = [data[m].get("stats_letterconf", {}).get("n_mh", 0) for m in models]
rep = [data[m].get("stats_letterconf", {}).get("n_repl", 0) for m in models]
cols = [COL[m] for m in models]
axc.barh(y - hc, sig, hc, color=cols, edgecolor="white", linewidth=0.4)
axc.barh(y, mh, hc, color=cols, edgecolor="white", linewidth=0.4, alpha=0.6)
axc.barh(y + hc, rep, hc, color=cols, edgecolor="white", linewidth=0.5, hatch="///", alpha=0.85)
for i in range(len(models)):
    axc.text(max(sig[i],rep[i])+0.8,y[i],f"{sig[i]}/{rep[i]}",va="center",fontsize=VAL,color="#333")
axc.set_yticks(y); axc.set_yticklabels(names); axc.invert_yaxis()
axc.set_xlim(0, max(sig + [1]) * 1.33); axc.set_xlabel("thief features")
axc.set_ylim(8.5,-1.5)
toplegend(axc, [Patch(facecolor="#888", label="BH"),
                Patch(facecolor="#888", alpha=0.6, label="MH"),
                Patch(facecolor="#888", hatch="///", edgecolor="white", label="rep.")], 3)
lab(axc, "c  enrichment (BH/rep.)"); clean(axc)

# ---- d: per-feature log-odds with 95% CI ----
rng = np.random.default_rng(0)
for i, m in enumerate(models):
    fs = data[m].get("stats_letterconf", {}).get("features", [])
    if not fs:
        axd.text(0.3, i, "none significant", va="center", fontsize=VAL, color="#666", style="italic")
        continue
    for f in fs:
        lo, hi = f["ci"]; yy = i + rng.uniform(-0.26, 0.26)
        axd.plot([lo, hi], [yy, yy], "-", lw=0.5, color=COL[m], alpha=0.5)
        axd.plot(f["log_odds"], yy, "o", ms=2.2, color=COL[m], mec="white", mew=0.2)
axd.axvline(0, lw=0.7, color="#888", ls="--")
axd.set_yticks(y); axd.set_yticklabels(names); axd.invert_yaxis()
axd.set_xlabel("log-odds, wrong vs base-correct")
lab(axd, "d  per-feature enrichment"); clean(axd)

# ---- e, f: held-out causal transfer ----
he = 0.36
sd_ = [data[m].get("transfer", {}).get("shortcut_mean_dmargin", 0) for m in models]
rd_ = [data[m].get("transfer", {}).get("random_mean_dmargin", 0) for m in models]
pw = [data[m].get("transfer", {}).get("p_wilcoxon_greater") for m in models]
axe.barh(y - he / 2, sd_, he, color=cols, edgecolor="white", linewidth=0.4)
axe.barh(y + he / 2, rd_, he, color=RAND, edgecolor="white", linewidth=0.4)
# The random control is |dmargin| < 0.02 on every model, which is invisible as a bar on this
# symlog axis. Mark each control with a short tick so the comparison is legible.
for i in range(len(models)):
    axe.plot([rd_[i]], [y[i] + he / 2], marker="|", ms=5.5, mew=1.1, color="#555", zorder=5)
for i in range(len(models)):
    # n.s. marks the one model whose gain is not significant against its random control
    tag = "%.2f" % sd_[i] + ("" if (pw[i] is not None and pw[i] < 0.05 / len(models)) else "  n.s.")
    axe.text(max(sd_[i], 0) * 1.3 + 0.012, y[i] - he / 2, tag, va="center", fontsize=VAL, color="#333")
axe.set_xscale("symlog", linthresh=0.1); axe.axvline(0, lw=0.7, color="#888")
axe.set_yticks(y); axe.set_yticklabels(names); axe.invert_yaxis()
axe.set_xlim(-0.05, 6); axe.set_xlabel("margin gain on held-out errors")
# No legend in e or f. Long bars sit at both ends of the axis, so every corner covers either a bar
# or a value label; the caption gives the key instead: coloured bars suppress that model's ten most
# frequent thieves, grey bars ten random features.
lab(axe, "e  held-out transfer"); clean(axe)
sf = [100 * data[m].get("transfer", {}).get("shortcut_flip_to_correct_rate", 0) for m in models]
rf = [100 * data[m].get("transfer", {}).get("random_flip_to_correct_rate", 0) for m in models]
axf.barh(y - he / 2, sf, he, color=cols, edgecolor="white", linewidth=0.4)
axf.barh(y + he / 2, rf, he, color=RAND, edgecolor="white", linewidth=0.4)
for i in range(len(models)):
    axf.plot([rf[i]], [y[i] + he / 2], marker="|", ms=5.5, mew=1.1, color="#555", zorder=5)
for i in range(len(models)):
    axf.text(sf[i] + 1.5, y[i] - he / 2, "%.0f" % sf[i], va="center", fontsize=VAL, color="#333")
axf.set_xlim(0, 100); axf.set_yticks(y); axf.set_yticklabels(names); axf.invert_yaxis()
axf.set_xlabel("held-out errors flipped (%)")
lab(axf, "f  flips against random"); clean(axf)

# ---- shared model legend, under the figure ----
fig.legend([Line2D([0], [0], color=COL[m], lw=2.4, marker="o", ms=3.2) for m in models], names,
           loc="lower center", ncol=4, fontsize=7.6, handlelength=1.8, columnspacing=1.2,
           bbox_to_anchor=(0.56, 0.005))

out = os.path.join(HERE, "F_validation.png")
__import__("publication_style").prepare_figure(fig, "F_validation"); fig.savefig(out, dpi=300, bbox_inches="tight",pad_inches=0.12, facecolor="white"); print("saved", out)
import shutil as _sh
tpl = os.path.join(os.path.dirname(HERE), "sn-article-template")
if os.path.isdir(tpl):
    _sh.copyfile(out, os.path.join(tpl, "F_validation.png")); print("synced -> sn-article-template/F_validation.png")
