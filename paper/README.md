# Paper: Interpretable Trajectory Segmentation for Longitudinal Data

LaTeX source for the SEAL case-study paper. Single-column arXiv-style `article`.

## Build on Overleaf

Upload the whole `paper/` folder. Overleaf compiles `main.tex` with pdfLaTeX and
BibTeX. The structure is standard and self-contained.

```
paper/
  main.tex              top-level document, \input's each section
  refs.bib              bibliography (natbib, plainnat)
  sections/
    abstract.tex
    introduction.tex
    related_work.tex
    problem_setting.tex
    algorithm.tex        method + preprocessing + temporal feature engineering
    case_studies.tex     co2, covid, worldbank, epa, and the CMIE health panel
    conclusion.tex
  figures/               vector PDFs, referenced by \includegraphics
    fig_pipeline.pdf
    fig_temporal_features.pdf
    fig_case_trajectories.pdf
    fig_tree_example.pdf
    fig_cmie_window.pdf
  make_figures.py        regenerates every figure from the run outputs
```

Required LaTeX packages are all standard on Overleaf (`amsmath`, `graphicx`,
`booktabs`, `algorithm`, `algpseudocode`, `natbib`, `hyperref`).

## Status

Four case studies (co2, covid, worldbank, epa) are complete with figures and rule
interpretation. The CMIE household-health case study carries the dataset, the
preprocessing, the design choices, and the temporal feature engineering, and its
results paragraph is a placeholder until the server run finishes. Reporting for
CMIE is aggregate and rule-level only, with no household-level record shown.

## Regenerating figures

```bash
python make_figures.py
```

Reads the run outputs in `seal_improved_runs/<dataset>/` and the local CMIE
`target_long.csv`, and writes the PDFs into `figures/`.

## Style

Formal, to the point, passive where it reads well. No colons and no em-dashes in
prose, by request. Feature names appear only inside figures, not in body text, so
there are no unescaped underscores to trip up LaTeX.
