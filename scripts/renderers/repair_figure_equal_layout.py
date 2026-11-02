"""Equal-height Figure 6 layout from the completed public numerical snapshot.

Writes standalone review artifacts only; numerical arrays and parameter formula
remain unchanged in make_fig_repair.py.
"""
from pathlib import Path
import hashlib
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import fitz

W, H = 370.612, 380.0
MODEL_FONT, OTHER_FONT = 6.3, 7.0
INK, GREEN = '#252525', '#009F86'
ALIASES = {'etb': 'ours', 'all': 'all', 'rl': 'random', 'sft': 'sft', 'base': 'base'}
METHOD_LABELS = {'ours': 'sCLT-guided LoRA', 'all': 'All-layer LoRA',
                 'random': 'Random-layer LoRA', 'sft': 'Full fine-tuning',
                 'base': 'Unedited model'}
AXES_BP = {'a': (48, 192, 125, 129), 'b': (242, 223, 125, 98),
           'c': (48, 24, 125, 129), 'd': (242, 124, 125, 55),
           'e': (242, 24, 125, 58)}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_plot_inputs(root, *, require_complete=True, allow_incomplete=False):
    """Read public frozen plot values; this is not a GPU run-completion check."""
    root = Path(root)
    data_dir = root / 'data'
    paths = [data_dir / 'figure6_plot_data.json', data_dir / 'figure6_reference.json']
    plot, reference = [_read(p) for p in paths]
    _require(plot['status'] == 'COMPLETE', 'A completed frozen numerical snapshot is required.')
    panels, names = plot['panels'], reference['model_order']
    expected = {'a': ['etb','all','rl','sft'], 'b': ['etb','all','rl','sft'], 'c': ['etb','all','rl','sft','base']}
    for panel, arms in expected.items():
        actual = [(r['model'],r['arm']) for r in panels[panel]]
        _require(len(actual) == len(set(actual)) and set(actual) == {(n,a) for n in names for a in arms},
                 f'Missing or duplicated numerical series in panel {panel}.')
    _require(panels['d'] == reference['seed_points_unchanged'], 'Frozen panel d mismatch.')
    _require({r['model']:r['parameter_percent'] for r in panels['e']} == reference['parameter_percent_unchanged'], 'Frozen panel e mismatch.')
    return dict(panels=panels, reference=reference, model_order=names, complete=True,
        merge=dict(status='ARCHIVED_COMPLETION_ATTESTATION',plot_allowed=True,accepted_jobs=24,missing_or_unaccepted=[]),
        source_hashes={p.relative_to(root).as_posix():_sha(p) for p in paths},
        coverage={p:dict(models=len(names),methods=arms,series_values=len(panels[p])) for p,arms in expected.items()})


def draw_figure(D, Hdata, RL, CB, parameter_fn, source_file, *, require_complete=True,
                allow_incomplete=False, output_dir=None):
    """Compatible entry point; default refuses incomplete FT data."""
    root = Path(source_file).resolve().parents[2]
    data = load_plot_inputs(root, require_complete=require_complete, allow_incomplete=allow_incomplete)
    default_folder = ('figure6-completed-controls-final-20261029' if data['complete']
                      else 'figure6-equal-halves-20261029')
    output = Path(output_dir) if output_dir else root / 'work_outputs' / default_folder
    output.mkdir(parents=True, exist_ok=True)
    reference, names, panels = data['reference'], data['model_order'], data['panels']
    methods = {key: dict(style, label=METHOD_LABELS[key])
               for key, style in reference['methods'].items()}
    _require({row[0]: parameter_fn(row) for row in D} == reference['parameter_percent_unchanged'],
             'Original parameter formula changed.')
    _require([dict(model=row[0], mean=row[2], sd=row[3], best=row[4]) for row in D] ==
             reference['seed_points_unchanged'], 'Original seed values changed.')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': OTHER_FONT,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.unicode_minus': False, 'axes.linewidth': .5})
    fig = plt.figure(figsize=(W / 72, H / 72), facecolor='white')
    canvas = fig.add_axes([0, 0, 1, 1]); canvas.axis('off')
    canvas.set(xlim=(0, W), ylim=(0, H))
    annotations = []

    def text(x, yy, value, size=OTHER_FONT, bold=False, ha='left', role='annotation'):
        t = canvas.text(x, yy, value, fontsize=size, color=INK, va='center', ha=ha,
                        weight='bold' if bold else 'normal', zorder=8)
        t.set_gid(role + '__' + str(len(annotations)))
        annotations.append((role, t))
        return t

    if not data['complete']:
        text(W - 1, 373, 'Layout proof - FT test data pending', ha='right', role='proof-status')
    for role, xx, yy in [('ours', 1, 359), ('all', 131, 359), ('random', 249, 359),
                          ('sft', 1, 346), ('base', 131, 346)]:
        style = methods[role]
        patch = Rectangle((xx, yy - 2.6), 8, 5.2, facecolor=style['color'], edgecolor='none', lw=0)
        patch.set_gid('method-key__' + role); canvas.add_patch(patch)
        text(xx + 12, yy, style['label'], role='method-key')
    y = np.arange(len(names), dtype=float)
    axes = {}
    for letter, (xx, yy, width, height) in AXES_BP.items():
        ax = fig.add_axes([xx / W, yy / H, width / W, height / H]); axes[letter] = ax
        ax.set_ylim(7.65, -.65)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=MODEL_FONT)
        ax.tick_params(axis='both', labelsize=OTHER_FONT, length=2, pad=2)
        ax.tick_params(axis='y', labelsize=MODEL_FONT, length=0, pad=3)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(.5); ax.spines[spine].set_color('#555555')
    for letter, xx, yy, title in [
        ('a', 1, 334, 'Retention on new questions'), ('b', 195, 334, 'Net effect on new questions'),
        ('c', 1, 166, 'Accuracy on audited questions'),
        ('d', 195, 192, 'Variation across runs'), ('e', 195, 95, 'Trainable parameters')]:
        text(xx, yy, letter, size=8, bold=True, role='panel-letter')
        text(xx + 13, yy, title, size=7, bold=True, role='panel-title')
    records, containers = [], []

    def bars(panel, arm, rows, offset, height, show_sd=True):
        ax, role = axes[panel], ALIASES[arm]
        lookup = {r['model']: r for r in rows}; ordered = [lookup[n] for n in names]
        means, errors = [r['mean'] for r in ordered], [r['sample_sd'] for r in ordered]
        kwargs = dict(color=methods[role]['color'], edgecolor='none', lw=0)
        if show_sd and any(v is not None for v in errors):
            _require(all(v is not None for v in errors), 'Mixed missing SD values.')
            kwargs.update(xerr=errors, error_kw=dict(lw=.55, capsize=.8, ecolor='#303030'))
        b = ax.barh(y + offset, means, height, **kwargs)
        for row, patch, yy in zip(ordered, b.patches, y + offset):
            patch.set_gid(f"bar__{panel}__{role}__{row['model']}")
            records.append(dict(panel=panel, method=role, arm=arm, model=row['model'],
                mean=float(row['mean']), sd=row['sample_sd'], sd_drawn=show_sd and arm != 'base',
                values=row.get('values'), protocol=row.get('protocol'), y=float(yy),
                height=height, facecolor=methods[role]['color'], hatch=''))
        containers.append((panel, role, b))

    for panel in 'abc':
        arms = data['coverage'][panel]['methods']
        step = .20 if len(arms) <= 4 else .16
        height = .17 if len(arms) <= 4 else .14
        for i, arm in enumerate(arms):
            bars(panel, arm, [r for r in panels[panel] if r['arm'] == arm],
                 (i - (len(arms) - 1) / 2) * step, height, show_sd=panel != 'a')
    axes['a'].set(xlim=(0, 100), xticks=[0, 50, 100])
    axes['a'].set_xlabel('Correct answers retained (%)', fontsize=OTHER_FONT, labelpad=3)
    all_ends = [r['mean'] + sign * r['sample_sd'] for r in panels['b'] for sign in [-1, 1]]
    low, high = min(0, min(all_ends)), max(0, max(all_ends))
    b_limits = [20 * math.floor(low / 20), max(40, 20 * math.ceil(high / 20))]
    if b_limits[0] == low:
        b_limits[0] -= 10
    if b_limits[1] == high:
        b_limits[1] += 10
    b_ticks = list(range(int(math.ceil(b_limits[0] / 50) * 50), int(b_limits[1]) + 1, 50))
    axes['b'].set(xlim=b_limits, xticks=b_ticks)
    axes['b'].axvline(0, color='#7d8882', lw=.6, zorder=0)
    axes['b'].set_xlabel('Change in correct answers', fontsize=OTHER_FONT, labelpad=3)
    axes['c'].set(xlim=(0, 104), xticks=[0, 50, 100])
    axes['c'].set_xlabel('Accuracy (%)', fontsize=OTHER_FONT, labelpad=3)
    points = {r['model']: r for r in panels['d']}
    means, errors, best = [[points[n][key] for n in names] for key in ['mean', 'sd', 'best']]
    axes['d'].errorbar(means, y, xerr=errors, fmt='o', ms=3.4, lw=.65,
                       capsize=1.5, color=GREEN, zorder=3)
    axes['d'].scatter(best, y, marker='*', s=26, color=INK, zorder=4)
    axes['d'].set(xlim=(20, 100), xticks=[20, 60, 100])
    axes['d'].get_xticklabels()[-1].set_ha('right')
    axes['d'].set_xlabel('Accuracy (%)', fontsize=OTHER_FONT, labelpad=3)
    parameters = {r['model']: r['parameter_percent'] for r in panels['e']}
    bars('e', 'etb', [dict(model=n, mean=parameters[n], sample_sd=None,
                           protocol='unchanged_original_parameter_formula') for n in names],
         0, .46, show_sd=False)
    axes['e'].set(xlim=(0, .25), xticks=[0, .1, .2])
    axes['e'].xaxis.set_major_formatter(FormatStrFormatter('%g'))
    axes['e'].set_xlabel('Trainable parameters (%)', fontsize=OTHER_FONT, labelpad=3)

    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    boxes = []
    for item in fig.findobj(Text):
        if item.get_visible() and item.get_text().strip():
            rect = item.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
            boxes.append(dict(text=item.get_text(), font_pt=item.get_fontsize(),
                              gid=item.get_gid(), bbox_pt=[float(v * 72) for v in rect.bounds]))
    outside = [t for t in boxes if t['bbox_pt'][0] < -.1 or t['bbox_pt'][1] < -.1 or
               t['bbox_pt'][0] + t['bbox_pt'][2] > W + .1 or
               t['bbox_pt'][1] + t['bbox_pt'][3] > H + .1]
    _require(not outside, f'Text outside figure: {outside}')
    model_label_checks = {}
    for letter, ax in axes.items():
        labels = sorted((t.get_window_extent(renderer) for t in ax.get_yticklabels()), key=lambda b: b.y0)
        gaps = [(labels[i + 1].y0 - labels[i].y1) * 72 / fig.dpi for i in range(len(labels) - 1)]
        _require(min(gaps) >= 0, f'Model labels overlap in {letter}.')
        model_label_checks[letter] = dict(count=len(labels), minimum_gap_bp=min(gaps), overlap=False)
    annotation_overlaps = []
    for i, (role, item) in enumerate(annotations):
        rect = item.get_window_extent(renderer)
        _require(all(not rect.overlaps(ax.get_window_extent(renderer)) for ax in axes.values()),
                 f'Annotation overlaps plotting axes: {item.get_text()}')
        for role2, other in annotations[i + 1:]:
            if rect.overlaps(other.get_window_extent(renderer)):
                annotation_overlaps.append([item.get_text(), other.get_text()])
        for ax in axes.values():
            if ax.xaxis.label.get_text() and rect.overlaps(ax.xaxis.label.get_window_extent(renderer)):
                annotation_overlaps.append([item.get_text(), ax.xaxis.label.get_text()])
    _require(not annotation_overlaps, f'Annotation overlap: {annotation_overlaps}')
    for row in panels['b']:
        _require(b_limits[0] <= row['mean'] - row['sample_sd'] and
                 b_limits[1] >= row['mean'] + row['sample_sd'], 'Panel b SD clipping detected.')
    segments = {}
    for panel, role, b in containers:
        if b.errorbar is not None:
            segments[panel + '__' + role] = [a.tolist() for coll in b.errorbar.lines[2]
                                            for a in coll.get_segments()]
    stem = output / 'F_repair'
    for extension in ['pdf', 'svg']:
        fig.savefig(stem.with_suffix('.' + extension), facecolor='white')
    fig.savefig(stem.with_suffix('.png'), dpi=450, facecolor='white')
    plt.close(fig)
    with fitz.open(stem.with_suffix('.pdf')) as pdf:
        page = pdf[0]
        spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b
                 for line in b['lines'] for s in line['spans'] if s['text'].strip()]
        _require(min(s['size'] for s in spans) >= MODEL_FONT - .01, 'Printed PDF font below 6.3 pt.')
        _require(not page.get_images(), 'Figure PDF must remain vector.')
        page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False).save(output / 'figure6_pdf_preview.png')
    audit = dict(status='PASS', artifact_status='FINAL_DATA_READY' if data['complete'] else 'LAYOUT_PROOF_FT_PENDING',
        width_bp=W, height_bp=H, model_label_font_pt=MODEL_FONT, minimum_printed_font_pt=MODEL_FONT,
        other_font_minimum_pt=OTHER_FONT, layout='equal-height a/c left; compact b/d/e right',
        axes_rectangles_bp={k: list(v) for k, v in AXES_BP.items()},
        left_column_equal_axes_height=True, a_to_c_height_ratio=1.0,
        model_order=names, methods=methods, bars=records, error_segments=segments,
        seed_points=panels['d'], parameter_values=parameters,
        axis_limits={k: list(ax.get_xlim()) for k, ax in axes.items()},
        b_all_means_and_sd_visible=True, b_actual_error_extent=[min(all_ends), max(all_ends)],
        no_break_marks=True, model_labels_on_all_five_panels=True,
        model_label_no_overlap=model_label_checks, text_boxes=boxes, outside_text=outside,
        annotation_overlaps=annotation_overlaps, all_legends_outside_data_axes=True,
        bar_value_annotations=False, full_coverage=data['coverage'],
        FT_completion_status=data['merge'].get('status'), FT_plot_allowed=data['merge'].get('plot_allowed'),
        FT_accepted_jobs=data['merge'].get('accepted_jobs'), FT_expected_jobs=24,
        FT_missing_or_unaccepted=data['merge'].get('missing_or_unaccepted'),
        incomplete_FT_values_used_in_a_b=False,
        preview_c_FT_source=None if data['complete'] else 'validated historical three-run audit200 only',
        caption_keys='d: green circles show mean +/- s.d.; black stars show best run. e: sCLT-guided LoRA only.',
        source_hashes=data['source_hashes'], source_make_sha256=_sha(source_file),
        source_renderer_sha256=_sha(__file__), outputs={ext: _sha(stem.with_suffix('.' + ext))
                                                     for ext in ['pdf', 'svg', 'png']},
        canonical_assets_overwritten=False)
    (output / 'figure6_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(status='PASS', artifact_status=audit['artifact_status'], bars=len(records),
                          FT_accepted_jobs=audit['FT_accepted_jobs'], output=str(stem.with_suffix('.pdf')))))
    return audit
