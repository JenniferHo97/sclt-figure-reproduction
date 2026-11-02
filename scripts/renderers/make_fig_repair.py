"""Figure 6 (formerly Figure 5): sCLT-guided LoRA repair, five-panel composite.
Reordered 2026-09-30 so the questions never used in the repair come first:
a correct answers kept on new questions; b net effect on new questions (both: all-layer LoRA,
random layers, sCLT); c repair versus all controls on audited questions;
d across-run variation; e trained-parameter footprint. Layout: a/c have equal
heights on the left, with b/d/e stacked on the right. The renderer reads audited
JSON and requires complete FT test results by default; explicit layout proofs
omit incomplete FT test comparisons. The literal tables below remain archived."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os
HERE=os.path.dirname(__file__)
# (name, base, repaired, seed s.d., best run, d_model, d_ff, n_params, n_sCLT_layers)
# IN-AUDIT protocol: scored on the same 200-question mixed audit the repair trains on, so this
# measures how completely the targeted errors are corrected, not generalisation -- panels a and b
# carry the disjoint-set result. Repaired = 3-seed mean, LoRA rank 16 on the layers the binomial
# sign test returns. Source fig5g/results/inaudit/ia16_<model>_{base,etb_sd42/43/44}.json,
# audit at fig5f/work/<model>/audit_mixed200.json.
D=[("Qwen-0.8B",      42.5,84.3,0.8,85.0, 1024,3072,0.8e9, 3),
   ("Qwen-2B-Base",   44.0,90.7,2.3,93.0, 2048,6144,2.0e9, 5),
   ("Qwen-2B-Inst",   50.5,88.2,1.0,89.0, 2048,6144,2.0e9, 3),
   ("R1-Distill-1.5B",21.0,65.0,2.6,67.0, 1536,8960,1.5e9, 2),
   ("Gemma-2-2B",     31.0,74.5,6.1,78.0, 2304,9216,2.6e9, 2),
   ("Gemma-3-1B",     24.5,37.3,1.2,38.0, 1152,6912,1.0e9, 4),
   ("OLMo-2-1B",      34.0,53.2,1.9,54.5, 2048,8192,1.0e9, 4),
   ("Phi-4-mini",     48.0,78.7,1.0,79.5, 3072,8192,3.8e9, 1)]
FUSED={"Phi-4-mini"}   # gate_up_proj is fused, so its LoRA shape differs
# (name, full-layer net, its s.d., full-layer retain, sCLT net, its s.d., sCLT retain)
# Disjoint 300-question test set, same configuration as panels a-c: LoRA rank 16, binomial-
# qualified layers, 3 seeds. Phi-4-mini's audit_phi4_test300.json was generated 2026-09-28;
# base accuracy on this set is 53.7%. Source fig5g/results/disjoint300/d300r16_*.
H=[("Qwen-0.8B",      -14.0,2.6,42.1,  +6.0,1.7,60.7),
   ("Qwen-2B-Base",   -15.3,10.8,66.4, +0.7,2.9,82.2),
   ("Qwen-2B-Inst",   -12.0,9.5,66.9, -14.0,2.6,77.2),
   ("R1-Distill-1.5B", -2.3,8.6,34.6,  +4.7,8.4,45.7),
   ("Gemma-2-2B",     -24.3,7.4,33.3,  +2.3,6.7,72.6),
   ("Gemma-3-1B",      -4.7,6.7,34.6,  +2.3,7.5,38.5),
   ("OLMo-2-1B",       +6.3,3.2,25.6,  +4.3,3.5,27.4),
   ("Phi-4-mini",     -17.0,7.5,75.8,  -1.0,1.7,94.6)]
# (name, random-layer net, its s.d., random-layer retain) on the same disjoint 300-question set.
# Count-matched random-layer control, 2026-09-30: same recipe, k layers drawn uniformly from ALL
# layers, one draw per seed (fig5g/work/randlayers.json). Source repro/code/rl_r16.sh ->
# fig5g/results/disjoint300/d300r16_<model>_rl_sd*.json; summary fig5g/results/rl_summary.json.
RL=[("Qwen-0.8B",       -3.7,2.1,47.4),
    ("Qwen-2B-Base",    -9.0,9.8,73.1),
    ("Qwen-2B-Inst",    -6.7,12.7,73.6),
    ("R1-Distill-1.5B", +5.3,7.6,16.5),
    ("Gemma-2-2B",      -8.7,12.9,61.1),
    ("Gemma-3-1B",      -2.7,7.2,26.8),
    ("OLMo-2-1B",       +4.3,4.0,24.1),
    ("Phi-4-mini",     -61.7,35.6,37.1)]
assert [r[0] for r in RL] == [h[0] for h in H]

# Same 200-question mixed audit and frozen control values as the archived plot.
CB=[("Qwen-2B-Base",44.0, 88.3,0.3, 78.7,5.8, 90.7,2.3),
    ("Qwen-2B-Inst",50.5, 86.0,5.4, 85.5,2.3, 88.2,1.0),
    ("Qwen-0.8B",   42.5, 73.0,1.7, 66.0,6.9, 84.3,0.8),
    ("Phi-4-mini",  48.0, 63.2,2.1, 88.3,2.3, 78.7,1.0),
    ("Gemma-2-2B",  31.0, 45.3,5.1, 46.8,7.6, 74.5,6.1),
    ("R1-Distill-1.5B",21.0, 24.2,0.6, 77.7,0.3, 65.0,2.6),
    ("OLMo-2-1B",   34.0, 30.0,1.3, 56.2,5.8, 53.2,1.9),
    ("Gemma-3-1B",  24.5, 31.7,0.3, 55.7,3.8, 37.3,1.2)]

def _lp(d):
    r=16; dm,df,npar,L=d[5],d[6],d[7],d[8]
    per = r*(2*dm+3*df) if d[0] in FUSED else 3*r*(dm+df)
    return 100*per*L/npar

if __name__ == "__main__":
    from repair_figure_layout import draw_figure
    draw_figure(D, H, RL, CB, _lp, source_file=__file__)
