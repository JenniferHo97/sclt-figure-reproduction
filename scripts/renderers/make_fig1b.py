# -*- coding: utf-8 -*-
"""Main Figure 3: frozen q0005 diagram and historical summaries.
Panel a is q0005. Panel b records 115 baseline-error token-patching cases;
it does not establish 115 successful repairs. Panel c is the archived q0177
single-case group-patching summary, not q0005 or a 115-case mean.
This release preserves the frozen manuscript plotting values and provides these scope
corrections in FIGURE_MAP.csv and README.md; it does not rerun model experiments.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch, RegularPolygon
import matplotlib.lines as ml
import matplotlib.patheffects as pe
import numpy as np, json, os
HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.path.abspath(os.path.join(HERE,"..","..","data"))

CASES={
 "q0005":dict(cat="q0005_feature_labels_cat.json", wrong="B", correct="A",
              evid=["fruity breath","dry mucosa","deep breathing","vomiting"],
              # option probabilities read from the logit nodes of the viewer graphs
              # (graph_viewer_pkg/data/qwen08_cases/medqa_q0005_A_B_{base,after}.json): base A .143 B .183 C .086 D .086; after A .945 B .022 C .012 D .020
              probs={"base":{"A":.14,"B":.18,"C":.09,"D":.09},"after":{"A":.95,"B":.02,"C":.01,"D":.02}}),
 "q0177":dict(cat="q0177_feature_labels_cat.json", wrong="C", correct="D",
              evid=["fluids","x-axis","volume","viscosity"],
              probs={"base":{"A":.14,"B":.10,"C":.26,"D":.10},"after":{"A":.02,"B":.01,"C":.01,"D":.97}}),
}
CASE_KEY=os.environ.get("FIG2_CASE","q0005")
assert CASE_KEY == "q0005", "Archived signed-edge rendering currently covers q0005 only"
CASE=CASES[CASE_KEY]
GRAPHS={tag:json.load(open(os.path.join(DATA, f"q0005_{tag}_graph.json"),encoding="utf-8")) for tag in ("base","after")}
CAT=json.load(open(os.path.join(DATA,CASE["cat"]),encoding="utf-8"))
WRONG,CORRECT=CASE["wrong"],CASE["correct"]

TEAL="#2E8B74"; GREEN="#4E9E7E"; RED="#C0604F"; GREY="#9AA0A6"; TAN="#C6A15B"; DIFF="#8FA6B8"; STEEL="#5f83a8"; INK="#25302b"
CATCOL={"medical":TEAL,"format":GREY,"other":TAN,"diffuse":DIFF}
# font sizes (source pt); on-page = source x ~0.77
FS_LET=15; FS_FTIT=11.5; FS_COL=9; FS_SUB=7.6; FS_FEAT=8.6; FS_EVID=8; FS_ANS=10.5; FS_ANSP=9
FS_NOTE=8.4; FS_TIT=10; FS_AX=9; FS_TICK=8.4; FS_VAL=9; FS_LEG=8.4; FS_CAP=8
plt.rcParams.update({"font.family":"DejaVu Sans","axes.linewidth":0.7})

m={}
for tag in ["base","after"]:
    for r in CAT[tag]:
        k=(r["layer"],r["local"]); m.setdefault(k,{"trig":r["trig"],"cat":r["cat"],"role":r["role"] or "","base":0,"after":0})
        m[k]["display_label"]=r["display_label"]; m[k][tag]=r["infl"]; m[k]["cat"]=r["cat"]; m[k]["trig"]=r["trig"]
        if r["role"]: m[k]["role"]=r["role"]
rows=[dict(layer=k[0],local=k[1],**v) for k,v in m.items()]
rows.sort(key=lambda r:-max(r["base"],r["after"]))
MID=rows[:7]
_thief=next((r for r in rows if r["role"]=="thief"),None)
if _thief and _thief not in MID: MID.append(_thief)
MEDWORDS=("influenza","hepatitis","vitamin","enzyme")
def flabel(r):
    if "display_label" in r: return r["display_label"]
    if r["role"]=="thief": return "shortcut"
    if r["cat"]=="diffuse": return "diffuse"
    if r["cat"]=="other": return "names"
    if r["cat"]=="medical":
        for t in r["trig"]:
            for w in MEDWORDS:
                if w in t.lower(): return w
        return "medical"
    t=[x for x in r["trig"] if x]
    if t==[":"]: return "‘:’ colon"
    if t and t[0].lower()=="answer": return "‘Answer’"
    return "punctuation"
def catsum(tag):
    s={"format":0,"other":0,"diffuse":0,"medical":0}
    for r in CAT[tag]: s[r["cat"]]=s.get(r["cat"],0)+r["infl"]
    return s
CB,CA=catsum("base"),catsum("after")
print("category influence  base:",{k:round(v,2) for k,v in CB.items()}," after:",{k:round(v,2) for k,v in CA.items()})

EVID=CASE["evid"]
PROBS=CASE["probs"]
GROUPS=[("content\npositions",54),("answer\nformat",-17),("answer\nposition",-44)]
PARA=[("strengthens\ncontent token",83),("full gold-word\nmatch",44)]
def clean(a):
    for s in ("top","right"): a.spines[s].set_visible(False)

def flow(ax,tag,title,correct):
    pr=PROBS[tag]; winner=CORRECT if correct else WRONG; infkey=tag
    xE,xF,xA=0.05,0.40,0.93
    yE=np.linspace(0.18,0.68,len(EVID)); yF=np.linspace(0.03,0.77,len(MID)); yA={"A":0.76,"B":0.55,"C":0.35,"D":0.14}
    bw=[max(0.20,0.0165*len(e)+0.03) for e in EVID]   # evidence box width follows the label length
    # Display only measured direct feature-to-logit edges. Clinical cues in the
    # left column are question context; no unmeasured cue-to-feature paths are drawn.
    graph=GRAPHS[tag]
    output={n['node_id']:n['clerp'].split('"')[1] for n in graph['nodes'] if n['feature_type']=='logit'}
    feature_y={(r['layer'],r['local']):yf for r,yf in zip(MID,yF)}
    visible={n['node_id']:feature_y[(int(n['layer']),int(n['node_id'].split('_')[1]))] for n in graph['nodes']
             if n['feature_type']=='cross layer transcoder' and (int(n['layer']),int(n['node_id'].split('_')[1])) in feature_y}
    for link in graph['links']:
        if link['source'] not in visible or link['target'] not in output: continue
        L=output[link['target']]; w=float(link['weight']); yf=visible[link['source']]; ya=yA[L]
        col=GREEN if L==CORRECT else RED if L==WRONG else '#808080'
        ax.add_patch(FancyArrowPatch((xF+0.05,yf),(xA-0.03,ya),arrowstyle='-',
                     connectionstyle='arc3,rad=0.04',lw=0.45+12*min(abs(w),0.35),
                     linestyle='-' if w>=0 else '--',color=col,alpha=0.65,zorder=1))
    for i,ye in enumerate(yE):
        ax.add_patch(FancyBboxPatch((xE+0.05-bw[i]/2,ye-0.028),bw[i],0.056,boxstyle="round,pad=0.004,rounding_size=0.02",fc="none",ec=TEAL,lw=1.2,zorder=3,clip_on=False))
        ax.text(xE+0.05,ye,EVID[i],ha="center",va="center",fontsize=FS_EVID,color=TEAL,zorder=4)
    for j,(r,yf) in enumerate(zip(MID,yF)):
        inf=r[infkey]; col=CATCOL.get(r["cat"],GREY); rad=0.007+0.036*inf
        shape=RegularPolygon((xF,yf),numVertices=3 if r["cat"]=="medical" else 4,radius=rad*1.25) if r["cat"] in ("medical","other") else Circle((xF,yf),rad)
        shape.set_facecolor(col if r["cat"]!="diffuse" else "white"); shape.set_edgecolor(RED if r["role"]=="thief" else col)
        shape.set_linewidth(1.8 if r["role"]=="thief" else 1.1); shape.set_alpha(.9); shape.set_zorder(3); ax.add_patch(shape)
        ax.text(xF+0.065,yf,flabel(r)+f" · L{r['layer']}",ha="left",va="center",fontsize=FS_FEAT,color="#222",zorder=6,path_effects=[pe.withStroke(linewidth=2.8,foreground="white")])
    for L,ya in yA.items():
        iswin=(L==winner); p=pr[L]; fc=(GREEN if L==CORRECT else RED) if iswin else "white"; ec=(GREEN if L==CORRECT else RED) if iswin else "#B7BDC3"
        rr=0.024+0.03*min(p,1)
        ax.add_patch(Circle((xA,ya),rr,facecolor=fc,edgecolor=ec,lw=1.7,zorder=4))
        ax.text(xA,ya,L,ha="center",va="center",fontsize=FS_ANS,fontweight="bold",color="white" if iswin else "#8a9096",zorder=5)
        ax.text(xA+0.055,ya,f"{p:.0%}",ha="left",va="center",fontsize=FS_ANSP,color=ec,fontweight="bold" if iswin else "normal",zorder=5)
    ax.text(xE+0.05,0.99,"clinical\nevidence",ha="center",va="top",fontsize=FS_SUB+0.5,color=TEAL,fontweight="bold")
    ax.text(xF+0.10,0.99,"sCLT features",ha="center",va="top",fontsize=FS_SUB+0.5,color="#6f767c",fontweight="bold")
    ax.text(xF+0.10,0.925,"(size ∝ influence)",ha="center",va="top",fontsize=FS_SUB+0.5,color="#9aa0a6")
    ax.text(xA,0.99,"answer",ha="center",va="top",fontsize=FS_SUB+0.5,color="#6f767c",fontweight="bold")
    ax.set_title(title,loc="left",fontsize=FS_FTIT,fontweight="bold",color=(RED if not correct else GREEN),pad=3)
    note=("Format features dominate." if not correct else "Format read-out weakens;\ncontent contributions increase.")
    ax.text(0.5,-0.04,note,ha="center",va="top",fontsize=FS_NOTE,color="#55605a",transform=ax.transAxes)
    ax.set_xlim(0,1.05); ax.set_ylim(-0.10,1.02); ax.axis("off")

fig=plt.figure(figsize=(6.7,8.0),dpi=300)
# 3 rows: (a) flows | legend band for a | (b,c,d) bars
outer=fig.add_gridspec(3,1,height_ratios=[1.20,0.6,1.00],hspace=0.32,left=0.105,right=0.98,top=0.95,bottom=0.12)
gtop=outer[0].subgridspec(1,2,wspace=0.16)
axL=fig.add_subplot(gtop[0,0]); axR=fig.add_subplot(gtop[0,1])
flow(axL,"base",f"Before repair → wrong answer {WRONG}",correct=False)
flow(axR,"after",f"After repair → correct answer {CORRECT}",correct=True)

# ---- legend for panel a, in its own band, pushed up toward a ----
ax_leg=fig.add_subplot(outer[1]); ax_leg.axis("off")
handles=[ml.Line2D([],[],marker='o',ls='',mfc=GREY,mec=GREY,ms=8,label="answer-format feature"),
         ml.Line2D([],[],marker='D',ls='',mfc=TAN,mec=TAN,ms=8,label="other non-medical"),
         ml.Line2D([],[],marker='o',ls='',mfc="white",mec=DIFF,ms=8,label="diffuse feature"),
         ml.Line2D([],[],marker='^',ls='',mfc=TEAL,mec=TEAL,ms=8,label="medical-concept feature"),
         ml.Line2D([],[],marker='o',ls='',mfc="white",mec=RED,ms=8,label="causal shortcut"),
         ml.Line2D([],[],color=GREEN,lw=3,label=f"to correct {CORRECT}"),
         ml.Line2D([],[],color=RED,lw=3,label=f"to original wrong {WRONG}"),
         ml.Line2D([],[],color="#666",lw=1.5,ls="-",label="positive edge"),
         ml.Line2D([],[],color="#666",lw=1.5,ls="--",label="negative edge")]
ax_leg.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.5,1.05),ncol=2,fontsize=FS_LEG,frameon=False,
              handletextpad=0.35,columnspacing=1.4,borderaxespad=0.0)

gbot=outer[2].subgridspec(1,3,wspace=0.62)
axa=fig.add_subplot(gbot[0,0]); axb=fig.add_subplot(gbot[0,1]); axc=fig.add_subplot(gbot[0,2])
FSX=FS_TICK-0.6

# ---- b: what the strengthened token is, over the 115 repaired cases (single 'relevant' bar) ----
axa.bar(0,73,width=0.62,color=GREEN)
axa.bar(1,9,width=0.62,color=TAN)
axa.bar(2,33,width=0.62,color=GREY)
axa.text(0,75,"73 (63%)",ha="center",va="bottom",color=GREEN,fontsize=FS_VAL,fontweight="bold")
axa.text(1,10.5,"9",ha="center",va="bottom",fontsize=FS_VAL,fontweight="bold")
axa.text(2,34.5,"33",ha="center",va="bottom",fontsize=FS_VAL,fontweight="bold")
axa.set_xticks([0,1,2]); axa.set_xticklabels(["match","part","other"],fontsize=FSX,rotation=25,ha="right")
axa.set_yticks([0,20,40,60]); axa.tick_params(axis="y",labelsize=FS_TICK); axa.set_ylim(0,100)
axa.set_ylabel("repaired cases (of 115)",fontsize=FS_AX)
axa.set_title("Lexical match\n(n = 115)",loc="left",fontsize=FS_TIT,fontweight="bold",pad=4); clean(axa)

# ---- c: causal patch by token group ----
gcol=[GREEN if v>0 else RED for _,v in GROUPS]
axb.bar(range(len(GROUPS)),[v for _,v in GROUPS],color=gcol,width=0.62)
axb.axhline(0,lw=0.7,color="#888"); axb.set_xticks(range(len(GROUPS)))
axb.set_xticklabels(["content","format","final"],fontsize=FSX,rotation=25,ha="right")
axb.tick_params(axis="y",labelsize=FS_TICK)
axb.set_ylabel("Repair effect\nrecovered (%)",fontsize=FS_AX); axb.set_ylim(-58,72)
for i,(g,v) in enumerate(GROUPS): axb.text(i,v+(3 if v>0 else -3),f"{v:+d}",ha="center",va="bottom" if v>0 else "top",fontsize=FS_VAL,fontweight="bold")
axb.set_title("Causal patch\nby group (L4)",loc="left",fontsize=FS_TIT,fontweight="bold",pad=4); clean(axb)

# ---- d: paraphrase ----
axc.bar(range(len(PARA)),[v for _,v in PARA],color=[GREEN,GREEN],width=0.56)
axc.set_xticks(range(len(PARA))); axc.set_xticklabels(["content\ntoken","gold-word\nmatch"],fontsize=FSX); axc.set_ylim(0,108)
axc.tick_params(axis="y",labelsize=FS_TICK)
axc.set_ylabel("Repaired cases (%)",fontsize=FS_AX)
for i,(p,v) in enumerate(PARA): axc.text(i,v+2,f"{v}%",ha="center",fontsize=FS_VAL,fontweight="bold",color=GREEN)
axc.set_title("Paraphrase\n(n = 72)",loc="left",fontsize=FS_TIT,fontweight="bold",pad=4); clean(axc)

# ---- panel letters, above/left of each title ----
axL.text(-0.05,1.17,"a",transform=axL.transAxes,fontsize=FS_LET,fontweight="bold",color=INK,va="top",ha="left")
for ax,l in [(axa,"b"),(axb,"c"),(axc,"d")]:
    ax.text(-0.34,1.34,l,transform=ax.transAxes,fontsize=FS_LET,fontweight="bold",color=INK,va="top",ha="left")

def reflow(figure):
    # This reserved legend band is re-applied after the final printed type size
    # is known; its symbols do not share a key with the lower statistical bars.
    legend_font = max(t.get_fontsize() for t in ax_leg.get_legend().get_texts())
    outer.set_height_ratios([1.20, max(0.60, 0.60*legend_font/9.5), 1.00])
    outer.update(hspace=0.32)

out=os.path.join(HERE,"F1b.png")
__import__("publication_style").prepare_figure(fig, "F1b", layout=reflow)
for _ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(HERE,"F1b."+_ext),dpi=300,facecolor="white",bbox_inches="tight",pad_inches=0.12)
print("saved",out,"| middle features:",[(flabel(r),r["layer"],r["cat"],round(r["base"],2),round(r["after"],2)) for r in MID])
import shutil as _sh, os as _os
for _ext in ("png", "pdf", "svg"):
    _dst=_os.path.join(_os.path.dirname(HERE),"sn-article-template","F1b."+_ext)
    _sh.copyfile(os.path.join(HERE,"F1b."+_ext),_dst)
print("synced -> sn-article-template/F1b.{png,pdf,svg}")
