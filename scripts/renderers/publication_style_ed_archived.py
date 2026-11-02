"""Make figure text measurable at the manuscript's actual printed width.

Only typography and non-colour encodings are changed. Data and limits are retained.
The font report records the rendered tight bounding box, including savefig padding.
"""
from pathlib import Path
import json
import matplotlib.colors as colors
from matplotlib.text import Text
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

HATCHES = ('', '///', '\\\\', 'xx', '..', '++', 'oo', '--', '**')
MARKERS = ('o', 's', '^', 'D', 'v', 'P', 'X', '*')

def prepare_figure(fig, name, width_pt=370.62):
    colour_keys = []
    preserve = set()
    for ax in fig.axes:
        bars = [p for c in ax.containers for p in getattr(c, 'patches', ())]
        if any(p.get_hatch() for p in bars):
            preserve.update(bars)
        for container in ax.containers:
            patches = getattr(container, 'patches', ())
            for patch in patches:
                c = colors.to_hex(patch.get_facecolor())
                if c not in colour_keys:
                    colour_keys.append(c)
    for patch in fig.findobj(Rectangle):
        if patch in [ax.patch for ax in fig.axes] or patch is fig.patch:
            continue
        c = colors.to_hex(patch.get_facecolor())
        if c in colour_keys:
            if patch not in preserve and not patch.get_hatch():
                patch.set_hatch(HATCHES[colour_keys.index(c) % len(HATCHES)])
            patch.set_edgecolor('#555555')
            patch.set_linewidth(max(0.35, patch.get_linewidth()))
    line_colours = []
    for line in [line for ax in fig.axes for line in ax.lines]:
        if len(line.get_xdata()) >= 3 and line.get_linestyle() not in ('None', 'none', ''):
            c = colors.to_hex(line.get_color())
            if c not in line_colours:
                line_colours.append(c)
            i = line_colours.index(c)
            line.set_marker(MARKERS[i % len(MARKERS)])
            line.set_markevery(max(1, len(line.get_xdata()) // 5))
            line.set_markersize(3.0)
            line.set_linestyle(('-', '--', '-.', ':')[i % 4])
    # Match pre-existing legend handles to the now distinguishable plotted series.
    for legend in fig.legends + [ax.get_legend() for ax in fig.axes if ax.get_legend()]:
        for line in legend.findobj(Line2D):
            c = colors.to_hex(line.get_color())
            if c in line_colours:
                i = line_colours.index(c)
                line.set_marker(MARKERS[i % len(MARKERS)])
                line.set_linestyle(('-', '--', '-.', ':')[i % 4])
                line.set_markersize(3.0)
    # Enlarging text can slightly widen the tight box, so solve to convergence.
    for _ in range(50):
        fig.canvas.draw()
        bbox = fig.get_tightbbox(fig.canvas.get_renderer())
        saved_width_pt = (bbox.width + 0.24) * 72  # <= 0.12-inch pad used below
        required = 7.22 * saved_width_pt / width_pt
        changed = False
        for text in fig.findobj(Text):
            if text.get_visible() and text.get_text().strip() and text.get_fontsize() < required - 0.002:
                text.set_fontsize(required)
                changed = True
        if not changed:
            break
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    saved_width_pt = (bbox.width + 0.24) * 72
    fonts = [x.get_fontsize() for x in fig.findobj(Text) if x.get_visible() and x.get_text().strip()]
    report = dict(figure=name, display_width_pt=width_pt, saved_width_pt=saved_width_pt,
                  minimum_source_font_pt=min(fonts), minimum_printed_font_pt=min(fonts)*width_pt/saved_width_pt,
                  bar_colours=len(colour_keys), line_colours=len(line_colours))
    assert report['minimum_printed_font_pt'] >= 7.19, report
    Path(__file__).with_name(name+'_font_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
