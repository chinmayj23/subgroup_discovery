# SEAL: code for the paper

SEAL groups longitudinal entities by the shape of their target trajectory and
describes each group with a compact rule over the entity descriptors. A single
randomized search fits the grouping and its rule together, scored by
`cost = rho + lambda * complexity`.

## Files

- `seal.py` — the method: pyramid trajectory representation and distance, the
  randomized clustering-to-tree search, rule extraction, and the diversity ranking.
- `run_panels.py` — runs SEAL on the four public panels (World Bank, OWID CO2,
  OWID COVID-19, EPA), writes `outputs/<panel>/membership.csv` and `rules.txt`,
  and renders the rule-card figure `figures/<panel>.pdf`.
- `baselines.py` — compares SEAL to the Cluster-then-Explain and EMM baselines on
  the shared objective (coherence `rho`, complexity `c`, cost), and prints the
  results table.
- `outputs/`, `figures/` — the results and figures reported in the paper.

## Data

The panels are built from public datasets, expected under `../seal_repo/data/`:
World Development Indicators, the Our World in Data CO2 and COVID-19 collections,
and the EPA Air Quality System. The target and its transformations are excluded
from the descriptors in every panel.

## Reproduce

```
python run_panels.py        # fit SEAL on all four panels and render figures
python baselines.py         # SEAL vs Cluster-then-Explain vs EMM
```

Dependencies: numpy, pandas, scipy, scikit-learn, matplotlib. A fixed random seed
makes every run reproducible.
