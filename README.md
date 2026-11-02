# sCLT figure reproduction

CPU plotting code and frozen numerical inputs for six main figures and six
Extended Data figures in the sCLT manuscript. Eleven figures can be redrawn;
Figure 1 is supplied as the preserved manuscript PNG reference image.
The figure snapshot is the 2026-10-29 submission candidate, packaged on
2026-11-02. Some renderer filenames retain historical numbering;
`FIGURE_MAP.csv` gives the current figure numbering.

Version 1.0.3 updates the release metadata after enabling Zenodo preservation.
Scientific values, plotting inputs and automatic rendering logic are unchanged
from version 1.0.2. Figure 1 remains a reference PNG only.

## Quick start

Python 3.11 or newer is recommended. The release was exercised with Python 3.13
on Windows; exact installed plotting versions are pinned in `requirements.txt`.

```sh
python -m venv .venv
# Activate the virtual environment using the command for your operating system.
python -m pip install -r requirements-lock.txt
python render_figures.py --check
python render_figures.py --output build
```

List or select figures:

```sh
python render_figures.py --list
python render_figures.py --only 2 6 --output build_selected
```

Outputs include PNG, PDF and SVG where available, plus numerical and typography
audit JSON. Each renderer executes in a temporary working copy. No model weights,
GPU, model service, API credentials or network access are needed to draw figures.
`--check` verifies public-file hashes, the 576 Figure 6 run-level values, their
192 mean/sample-SD groups and all 96 edited-method series in panels a-c.

`requirements.txt` pins the four direct plotting dependencies;
`requirements-lock.txt` also pins the transitive dependencies from the accepted
fresh installation. On 2 November 2026, a new virtual environment was installed
from the direct requirements and all eleven automatic figures were redrawn.
All twelve reference comparisons, including the Figure 1 reference image,
matched at the pixel level under the recorded Windows fonts. The acceptance
record is `verification/clean_install_acceptance.json`.
The local CPU acceptance and reference comparisons are recorded in
`verification/plot_reproduction_report.json`. The verified run used the existing
pinned environment; a fresh dependency installation was not part of that check.

## Typography and languages

English Figure 2 is the default. Arial is used by several archived layouts;
Matplotlib uses an installed fallback if Arial is absent. Numerical inputs stay
the same, while font substitution can change text placement and pixels. The
original Microsoft fonts are not redistributed. Optional Chinese Figure 2 can
be drawn using a CJK font you already have permission to use:

```sh
python render_figures.py --only 2 --languages en zh --cjk-font /path/to/cjk-font.ttf
```

## Repository contents

- `render_figures.py`: one CPU entry point and integrity/numerical checks.
- `scripts/renderers/`: plotting functions and the two archived style helpers.
- `data/`: minimum numerical arrays, displayed feature labels and graph edges.
- `data/figure6_runs_numeric.csv`: three archived run values for each model,
  method, cohort and metric; these support the plotted means and sample SDs.
- `data/SOURCES.json`: archived-source and public-file hashes, with adaptations.
- `data/PLOT_MANIFEST.json`: integrity hashes of the released plotting payload.
- `FIGURE_MAP.csv`: figure number, renderer, inputs and evidence scope.
- `figures/reference/`: preserved manuscript figures used for comparison.
- Main Figure 1 is provided only as a reference PNG image.

## What is reproduced

This repository reconstructs figures from frozen plotted values. It does not
retrain dictionaries, reproduce GPU experiments or establish historical model
identities beyond the recorded source evidence. Input minimization removes full
MedQA questions, textbook passages, training activation examples, input token
IDs, service configurations and private filesystem paths. Figure 2 keeps all
222 input-embedding positions with anonymous position-based IDs; the displayed
features and actual edge weights are unchanged. Figure 3 retains only displayed
feature labels and the measured graph fields used by its renderer.

Important interpretation details accompany the unchanged historical values:

- Figure 3a is q0005; 3b describes 115 baseline-error token-patching records,
  rather than proving that all 115 became correct. Figure 3c is the separately
  archived q0177 single-case token-group summary, not q0005 or a cohort average.
  Some preserved figure labels use the older wording; this scope note governs
  their interpretation. Exact adapter/run identity for parts of that archive
  remains incomplete.
- Figure 3's historical medical/format/other feature labels come from lexical
  and candidate semantic tables. Reproducing those labels is not a new clinical
  or mechanistic validation of them.
- Figure 6a-c include all four edited methods, with the unedited baseline also
  in c. Three archived run labels are not uniformly controlled independent RNG
  replications. The Qwen-0.8B and Phi-4 audit item sets differ from those of the
  other six models. The 300-question evaluation is held out from the repair
  step; complete exclusion from every earlier dictionary-selection or training
  stage has not been established.
- Figure 6d-e preserve earlier rounded run summaries and the original parameter
  formula. ED4 preserves its historical 182-item paraphrase evaluation. ED5 has
  six models for task-native PubMedQA repair and ED6 seven for MedEditBench
  transfer; omitted model results are not synthesized.
  ED6's Phi-4 endpoints can be redrawn from the frozen summary, while their
  original item-level source outputs have not been recovered.
- Figure 1 is copied from the preserved manuscript PNG as a reference image.
  The hand-edited schematic source is not part of this release.

Model/dataset source links and the complete manuscript methods should be read
alongside these plotting inputs. The repository's code licence does not change
the terms of third-party models, datasets, dependencies or fonts.

## Licence and citation

See `LICENSE` for author-owned code and `THIRD_PARTY_NOTICES.md` for dependency
and source scope. Cite the fixed release using the repository's citation metadata
and its archived DOI once registered.

Official repository: <https://github.com/JenniferHo97/sclt-figure-reproduction>.
