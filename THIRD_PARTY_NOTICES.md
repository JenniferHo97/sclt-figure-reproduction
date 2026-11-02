# Third-party software and source scope

The MIT licence in `LICENSE` applies to the authors' own plotting code. The
following dependencies are installed separately, not bundled as vendored code:

| Dependency | Source | Upstream licence |
|---|---|---|
| NumPy | https://numpy.org/ | BSD-3-Clause |
| Matplotlib | https://matplotlib.org/ | Matplotlib licence, based on the PSF licence |
| SciPy | https://scipy.org/ | BSD-3-Clause |
| PyMuPDF / MuPDF | https://pymupdf.readthedocs.io/ | GNU AGPL-3.0 or a separately obtained commercial licence |

Consult the exact installed distribution's notices for complete dependency
terms. The repository's MIT choice does not relicense those dependencies.

Fonts are obtained from the user's operating system or chosen by Matplotlib.
Microsoft Arial and Microsoft YaHei are not redistributed here. A user-provided
CJK font can be selected for the optional Chinese Figure 2; its terms remain
with its provider.

The numerical summaries and diagram layouts were produced in the sCLT study.
Only minimal frozen plotting data are included. Full MedQA/MedEditBench/PubMedQA
question text, textbook excerpts, training activation examples, pretrained model
weights and sCLT checkpoints are not redistributed in this plotting release.

Relevant upstream resources, with their own applicable terms:

- MedQA: https://github.com/jind11/MedQA
- PubMedQA: https://pubmedqa.github.io/
- sCLT study model families: Qwen, Gemma, OLMo, DeepSeek-R1-Distill and Phi.
  No model weights or upstream model software are included.
- Historical full-width CLT comparison identifier:
  `mntss/clt-gemma-2-2b-426k` on Hugging Face; no checkpoint is bundled.

Displayed medical terms, feature IDs and brief generic clinical cue labels are
included as parts of the authors' analytical figures. Their presence does not
grant rights to reproduce the original benchmark's complete question corpus.

`data/SOURCES.json` records archived source hashes and the public minimization
steps. `README.md` and `FIGURE_MAP.csv` describe evidence limits, including the
distinction between reproducing a figure and rerunning its original experiment.
