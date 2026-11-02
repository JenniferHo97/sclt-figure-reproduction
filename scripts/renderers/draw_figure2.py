"""Reflow the frozen Figure 2 into two side-by-side vector panels.

Only presentation changes: all graph edges, input embeddings and native-model
response values are read from frozen records. This program runs no model/API.
"""
from pathlib import Path
import hashlib
import json
import math

import fitz
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from matplotlib.text import Text
from matplotlib.ticker import FormatStrFormatter

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
GRAPH = ROOT / 'data/figure2_graph.json'
RESPONSE = ROOT / 'data/figure2_native_response.json'
D = json.loads(GRAPH.read_text(encoding='utf8'))
P = json.loads(RESPONSE.read_text(encoding='utf8'))
assert hashlib.sha256(GRAPH.read_bytes()).hexdigest() == 'b8e0529513506a92be440f3e02184f14759aeadb7e59ecabc12a7199054cd948'
assert hashlib.sha256(RESPONSE.read_bytes()).hexdigest() == '554e50155d2b4a6d3a0894b12c41ab60b0921ff08bbd14bfe82637b3e67eab2b'

W, H = 370.612, 200.
# All five original main figures are placed at the same ~370.61pt text width.
# Reflow at that final size; reserve more width for the intervention panel.
GW, GAP = 212., 12.
BX = GW + GAP
GX, GY = GW / 226., H / 252.
GREEN, RED, GREY, INK = '#4f9f85', '#c3614f', '#969da3', '#26312f'
import os
ZH = FontProperties(fname=os.environ['SCLT_CJK_FONT']) if os.environ.get('SCLT_CJK_FONT') else FontProperties(family='Microsoft YaHei')
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7.2, 'pdf.fonttype': 42,
                     'svg.fonttype': 'none', 'axes.unicode_minus': False})

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def draw(lang):
    fig = plt.figure(figsize=(W / 72, H / 72), facecolor='white')
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set(xlim=(0, W), ylim=(0, H)); canvas.axis('off')
    objects = []
    def text(x, y, value, size=7.2, ha='left', color=INK, bold=False, cjk=False,
             white=False, role='graph-label'):
        if role not in ['panel-letter', 'panel-title', 'response-key']:
            x, y = x * GX, y * GY
        t = canvas.text(x, y, value, fontsize=size, ha=ha, va='center', color=color,
                        fontweight='bold' if bold else 'normal',
                        fontproperties=ZH if lang == 'zh' and cjk else None,
                        bbox=dict(facecolor='white', edgecolor='none', pad=.3) if white else None,
                        zorder=8)
        t.set_gid(role + '__' + str(len(objects)))
        objects.append((role, t))
        return t

    # Identical font, weight and vertical alignment for both panel headings.
    for x, letter, title in [(1, 'a', '解释与发现' if lang == 'zh' else 'Explain & discover'),
                              (BX + 1, 'b', '干预与验证' if lang == 'zh' else 'Intervene & validate')]:
        text(x, H - 13, letter, size=11, bold=True, role='panel-letter').set_va('baseline')
        text(x + 15, H - 13, title, size=9, bold=True, cjk=True, role='panel-title').set_va('baseline')
    if lang == 'zh':
        text(16, 226, '局部归因', color='#68766f', cjk=True, role='attribution-unit')
        text(48, 226, 'Aᵢ (logit)', color='#68766f', role='attribution-unit')
    else:
        text(16, 226, 'Local attribution, Aᵢ (logit)', color='#68766f', role='attribution-unit')

    original_xy = {'1_3798_160': (45, 58), '11_6070_221': (150, 106),
          '21_4907_221': (77, 160), '21_6845_221': (163, 160),
          'logit_C': (87, 201), 'logit_B': (183, 201)}
    xy = {node: (x * GX, y * GY) for node, (x, y) in original_xy.items()}
    fs = D['core_features']
    radii = {f['node_id']: 3.4 * math.sqrt(abs(f['local_C_minus_B']) / .09818435460329056) for f in fs}
    radii.update(logit_C=7, logit_B=7)
    for y in [84, 134, 187]:
        canvas.plot([23 * GX, GW - 2], [y * GY, y * GY], color='#e4e8e9', lw=.35, zorder=0)
    for label, y in [('L1', 77), ('L11', 124), ('L21', 181),
                     ('输出' if lang == 'zh' else 'Output', 201)]:
        text(1, y, label, color='#78848a', cjk=True, role='layer-label')

    # Every original input embedding is present, even in this compact excerpt.
    embs = D['all_embedding_nodes']
    assert len(embs) == 222
    emb_edges = list(D['optional_print_embedding_edges_top1_per_core'].values())
    selected = {e['source'] for e in emb_edges}
    input_x, input_width = 24. * GX, 198. * GX
    for i, e in enumerate(embs):
        col, row = i % 111, i // 111
        x, y = input_x + col * input_width / 111, (9.4 - row * 4.1) * GY
        tile = Rectangle((x, y), input_width / 111 - .13, 3.1 * GY,
                         facecolor=GREEN if e['position'] == 160 else '#7f878d' if e['node_id'] in selected else '#cdd3d6',
                         edgecolor='none', zorder=2)
        tile.set_gid('embedding__' + e['node_id']); canvas.add_patch(tile)
        xy[e['node_id']] = (x + (input_width / 111 - .13) / 2, y + 1.55 * GY)
    text(24, 21, '输入词元（222个）' if lang == 'zh' else 'Input tokens (n=222)',
         color='#78848a', cjk=True, white=True, role='input-count')

    edges = []
    def arrow(e, color, lw, curvature, kind, alpha):
        a, b = e['source'], e['target']
        edge = FancyArrowPatch(xy[a], xy[b], arrowstyle='-|>', mutation_scale=4.6,
                               shrinkA=radii.get(a, 1) + 1, shrinkB=radii.get(b, 1) + 1,
                               connectionstyle=f'arc3,rad={curvature}', lw=lw, color=color,
                               linestyle='-' if e['weight'] > 0 else (0, (2.5, 1.7)),
                               alpha=alpha, zorder=3 if kind == 'answer_link' else 1)
        edge.set_gid(kind + '__' + a + '__' + b); canvas.add_patch(edge)
        edges.append(dict(source=a, target=b, weight=e['weight'], kind=kind))
    for e in emb_edges:
        arrow(e, '#adb8bd', .45, .06, 'embedding_link', .7)
    for e in D['exact_core_to_core_edges']:
        curv = -.18 if e['source'] == '1_3798_160' else -.20
        arrow(e, '#a6b1b6', .5, curv, 'feature_link', .8)
    maxw = max(abs(e['weight']) for e in D['exact_core_to_B_C_edges'])
    for e in D['exact_core_to_B_C_edges']:
        curv = -.34 if e['source'] == '21_6845_221' and e['target'] == 'logit_B' else .05 if e['target'] == 'logit_C' else -.18
        arrow(e, GREEN if e['target'] == 'logit_C' else RED,
              .45 + 1.65 * abs(e['weight']) / maxw, curv, 'answer_link', .92)

    zh = {7278: '白细胞介素／\n干扰素名称片段', 122542: '作答提示',
          223895: '数字／\n答案位置', 225833: '标点／\n作答边界'}
    en = {7278: 'Interleukin /\ninterferon fragments', 122542: 'Answer prompt',
          223895: 'Numbers /\nanswer slots', 225833: 'Punctuation /\nanswer boundary'}
    labels = zh if lang == 'zh' else en
    placements = {7278: (57, 'left', 80, 60, 39),
                  122542: (139, 'right', 127, 113, 99),
                  223895: (66, 'right', 181, 162, 142),
                  225833: (174, 'left', 181, 162, 142)}
    for f in fs:
        g, node = f['g'], f['node_id']
        color = GREEN if g == 7278 else GREY
        circle = Circle(xy[node], radii[node], facecolor='white' if g in (7278, 122542) else '#e3e6e8',
                        edgecolor=color, lw=1.1, zorder=6)
        circle.set_gid('feature__' + node); canvas.add_patch(circle)
        tx, align, idy, labely, valuey = placements[g]
        text(tx, idy, f'F{g}', ha=align, bold=True, white=True, role='feature-id')
        text(tx, labely, labels[g], ha=align, color=color if g == 7278 else INK,
             cjk=True, white=True, role='feature-semantics')
        text(tx, valuey, f'Aᵢ {f["local_C_minus_B"]:+.3f}', ha=align, white=True,
             role='feature-attribution')
    for letter in ['C', 'B']:
        x, y = xy['logit_' + letter]
        color = GREEN if letter == 'C' else RED
        circle = Circle((x, y), 7, facecolor=color if letter == 'C' else 'white',
                        edgecolor=color, lw=1.1, zorder=6)
        circle.set_gid('answer__logit_' + letter); canvas.add_patch(circle)
        x, y = original_xy['logit_' + letter]
        text(x, y, letter, size=9, ha='center', color='white' if letter == 'C' else RED, bold=True,
             role='answer-letter')
        text(x, y + 14, ('正确项' if letter == 'C' else '竞争项') if lang == 'zh' else
             ('correct' if letter == 'C' else 'competitor'), ha='center', color=color, cjk=True, role='answer-role')

    # Native-model response, displayed on the same row as the attribution graph.
    doses = np.array(P['dose'])
    target = np.array(P['target_delta'])
    ax = fig.add_axes([(BX + 34) / W, 40 / H, (W - BX - 39) / W, (H - 98) / H])
    ax.fill_between(doses, P['random_min'], P['random_max'], color='#dce1df', alpha=.8)
    ax.plot(doses, P['random_median'], color='#8a9690', lw=.7)
    ax.plot(doses, target, 'o-', ms=3.4, lw=1.2, color=GREEN, zorder=4)
    ax.axhline(0, color='#aab3af', lw=.5, zorder=0)
    ax.axvline(0, color='#cbd2ce', lw=.5, ls=':', zorder=0)
    for d, y, c in zip(doses, target, P['actual_free_generation_choices']):
        ax.annotate(c, (d, y), xytext=(0, 5), textcoords='offset points', ha='center',
                    fontsize=7.2, color=GREEN if c == 'C' else RED, weight='bold')
    ys = np.r_[P['random_min'], P['random_max'], target, 0.]
    span = float(np.ptp(ys))
    ax.set_ylim(float(ys.min()) - .12 * span, float(ys.max()) + .16 * span)
    ax.set_xlim(-1.08, 1.08)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-.2, 0, .2]); ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.tick_params(axis='both', labelsize=7.2, length=2, pad=2)
    kw = dict(fontproperties=ZH) if lang == 'zh' else dict(fontfamily='Arial')
    ax.set_xlabel('干预增量 d' if lang == 'zh' else 'Intervention increment, d', fontsize=7.2, labelpad=4, **kw)
    ax.set_ylabel('分数差变化 Δm（logit）' if lang == 'zh' else 'Margin change, Δm (logit)',
                  fontsize=7.2, labelpad=3, **kw)
    ax.spines[['top', 'right']].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(.5); ax.spines[spine].set_color('#a7b2ac')
    # Compact shared key; the baseline, dose and answer-letter definitions belong
    # in the caption, as requested, rather than a footer inside the figure.
    canvas.plot([BX + 37, BX + 49], [H - 30, H - 30], color=GREEN, marker='o', markevery=[1], ms=3.0, lw=1.2)
    text(BX + 55, H - 30, 'F7278', role='response-key')
    canvas.add_patch(Rectangle((BX + 37, H - 45.5), 12, 5, facecolor='#dce1df', edgecolor='none'))
    canvas.plot([BX + 37, BX + 49], [H - 43, H - 43], color='#8a9690', lw=.7)
    text(BX + 55, H - 43, '匹配随机方向' if lang == 'zh' else 'Matched random', cjk=True, role='response-key')

    # Geometric and source-data audit for every element we reflowed.
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    boxes = []
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip():
            rect = t.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
            boxes.append(dict(text=t.get_text(), font=t.get_fontsize(), bbox_pt=[v * 72 for v in rect.bounds]))
    stem = OUT / f'figure2_horizontal_{lang}'
    for extension in ['svg', 'pdf']:
        fig.savefig(stem.with_suffix('.' + extension), facecolor='white')
    plt.close(fig)
    with fitz.open(stem.with_suffix('.pdf')) as doc:
        p = doc[0]
        p.get_pixmap(matrix=fitz.Matrix(4, 4), alpha=False).save(stem.with_suffix('.png'))
        spans = [s for b in p.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans'] if s['text'].strip()]
        assert min(s['size'] for s in spans) >= 7.19
        assert not p.get_images()
        assert all(f'F{g}' in p.get_text() for g in [7278, 122542, 223895, 225833])
        assert 'Baseline' not in p.get_text() and '原始C' not in p.get_text()
        assert all(x['bbox_pt'][0] >= -.1 and x['bbox_pt'][1] >= -.1 and
                   x['bbox_pt'][0] + x['bbox_pt'][2] <= W + .1 and x['bbox_pt'][1] + x['bbox_pt'][3] <= H + .1 for x in boxes), boxes
        minfont = min(s['size'] for s in spans)
    expected = D['exact_core_to_core_edges'] + D['exact_core_to_B_C_edges'] + emb_edges
    assert sorted((e['source'], e['target'], e['weight']) for e in edges) == sorted((e['source'], e['target'], e['weight']) for e in expected)
    assert len(edges) == 17
    audit = dict(status='PASS', language=lang, horizontal_two_panels=True, width_pt=W, height_pt=H,
                 minimum_PDF_font_pt=minfont, raster_images=0, panel_title_font_pt=9,
                 panel_titles_bold=True, panel_title_baseline_y_pt=H - 13, all_embeddings=222,
                 graph_panel_width_pt=GW, panel_gap_pt=GAP, response_panel_width_pt=W - BX,
                 response_axes_width_pt=W - BX - 39,
                 width_reference='five original main figures at textwidth, measured in the compiled PDF',
                 input_token_count_excludes_generated_answer=True, actual_drawn_edges=edges,
                 input_embedding_ids=[e['node_id'] for e in embs], feature_radii=radii,
                 text_boxes=boxes, node_positions=xy,
                 attribution_notation='Aᵢ (logit)', response_notation='Δm (logit)',
                 x_axis='dimensionless decoder-contribution increment d', baseline_explanation_in_caption=True,
                 graph_data_sha256=sha(GRAPH), native_response_data_sha256=sha(RESPONSE),
                 native_response_values=P, pdf_sha256=sha(stem.with_suffix('.pdf')),
                 png_sha256=sha(stem.with_suffix('.png')))
    (OUT / f'figure_audit_{lang}.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf8')
    return audit

if __name__ == '__main__':
    audits = {lang: draw(lang) for lang in os.environ.get('SCLT_RENDER_LANGUAGES', 'en').split(',')}
    (OUT / 'verification.json').write_text(json.dumps(dict(status='PASS', figures=audits,
          public_plot_inputs_not_modified_during_render=True, measured_edges_and_response_values_preserved=True, model_or_API_runs=0),
          ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({lang: {k: a[k] for k in ['width_pt', 'height_pt', 'minimum_PDF_font_pt', 'raster_images', 'pdf_sha256']} for lang, a in audits.items()}, ensure_ascii=False))
