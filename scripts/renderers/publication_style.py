"""Make figure text measurable at the manuscript's actual printed width.

Only typography is adjusted here. Semantic encodings belong to each source figure:
explicit hatches, line styles and markers are preserved without inventing new ones.
The font report records the rendered tight bounding box, including savefig padding.
"""
from pathlib import Path
import json
import matplotlib.colors as colors
from matplotlib import rcParams
from matplotlib.text import Text
def prepare_figure(fig, name, width_pt=370.62, layout=None):
    """Ensure the printed minimum type size, reflowing after every size change.

    ``layout(fig)`` may reserve legend bands or rerun a figure-specific layout.
    It is called before measurement and again after typography has converged.
    Existing callers remain supported; no bar/line/marker encoding is changed.
    """
    # Keep exported text editable and embedded in the vector deliverables.
    rcParams["pdf.fonttype"] = 42
    rcParams["svg.fonttype"] = "none"
    colour_keys = []
    for ax in fig.axes:
        for container in ax.containers:
            patches = getattr(container, 'patches', ())
            for patch in patches:
                c = colors.to_hex(patch.get_facecolor())
                if c not in colour_keys:
                    colour_keys.append(c)
    line_colours = []
    for line in [line for ax in fig.axes for line in ax.lines]:
        if len(line.get_xdata()) >= 3 and line.get_linestyle() not in ('None', 'none', ''):
            c = colors.to_hex(line.get_color())
            if c not in line_colours:
                line_colours.append(c)
    # Enlarging text can slightly widen the tight box, so solve to convergence.
    for _ in range(50):
        if layout is not None:
            layout(fig)
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
    if layout is not None:
        layout(fig)
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    saved_width_pt = (bbox.width + 0.24) * 72
    fonts = [x.get_fontsize() for x in fig.findobj(Text) if x.get_visible() and x.get_text().strip()]
    report = dict(figure=name, display_width_pt=width_pt, saved_width_pt=saved_width_pt,
                  minimum_source_font_pt=min(fonts), minimum_printed_font_pt=min(fonts)*width_pt/saved_width_pt,
                  bar_colours=len(colour_keys), line_colours=len(line_colours),
                  semantic_encoding_policy="explicit_source_styles_only",
                  layout_reflow=layout is not None)
    assert report['minimum_printed_font_pt'] >= 7.19, report
    Path(__file__).with_name(name+'_font_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
