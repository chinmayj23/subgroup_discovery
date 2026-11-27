"""Run SEAL on the four public longitudinal panels and render the rule-card figures.

Panels: World Bank (output per person), OWID CO2 (emissions per capita),
OWID COVID-19 (weekly deaths per million), EPA (monthly PM2.5).

Descriptors follow the paper. A static covariate contributes its value, a
categorical covariate one indicator per level, and a time-varying covariate its
value at a few reference times plus its net change over the window. The target and
its transformations are excluded from the descriptors.

Outputs per panel: outputs/<ds>/membership.csv, outputs/<ds>/rules.txt, and the
figure figures/<ds>.pdf.
"""
from __future__ import annotations
import re
import math
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seal import run_seal

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "outputs"
FIG = HERE / "figures"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]

PANELS = {  # ds -> (csv, target, title, ylabel, xlabel)
    "worldbank": ("worldbank/worldbank_preprocessed.csv", "target_gdp",
                  "World Bank", "GDP per capita (1990 = 1)", "year"),
    "co2":       ("co2/co2_preprocessed.csv", "target_co2",
                  "CO2 emissions", "CO2 emissions per capita (t)", "year"),
    "covid":     ("covid/covid_preprocessed.csv", "target_deaths_per_million",
                  "COVID mortality", "COVID deaths per million", "week"),
    "epa":       ("epa/epa_preprocessed.csv", "target_pm25",
                  "EPA PM2.5", "PM2.5 (ug/m3)", "month"),
}
FEATURE_YEARS = (1990, 2000, 2010, 2020)
CHANGE_SPAN = {"worldbank": (1990, 2023), "co2": (1990, 2022)}
PARAMS = dict(n_candidates=20000, k_range=[2, 8], tree_depths=[1, 2, 3, 4],
              leaf_fractions=[0.02, 0.30], lambda_complexity=0.005,
              diversity_pool=500, ari_max=0.8, seed=0)


def load(ds):
    csv, target, *_ = PANELS[ds]
    df = pd.read_csv(DATA / csv, dtype={"entity": str} if ds == "epa" else None)
    traj = df.pivot(index="entity", columns="time", values=target).sort_index()
    inputs = [c for c in df.columns if c not in ("entity", "time", target)]
    if ds in ("worldbank", "co2"):
        wide = df.pivot(index="entity", columns="time", values=inputs).sort_index()
        years = list(traj.columns)
        feat = {}
        for cov in inputs:
            for y in FEATURE_YEARS:
                yy = y if y in years else min(years, key=lambda c: abs(c - y))
                feat[f"{cov}_{y}"] = wide[(cov, yy)]
            a, b = CHANGE_SPAN[ds]
            aa = a if a in years else min(years)
            bb = b if b in years else max(years)
            feat[f"{cov}_change_{a}_{b}"] = wide[(cov, bb)] - wide[(cov, aa)]
        X = pd.DataFrame(feat, index=traj.index)
        X = X.fillna(X.median())
    elif ds == "covid":
        X = df.groupby("entity")[inputs].first().loc[traj.index]
        X = X.fillna(X.median())
    else:  # epa: numeric site covariates + one-hot categoricals
        raw = df.groupby("entity")[inputs].first().loc[traj.index]
        cat = ["land_use", "location_setting", "state"]
        num = raw.drop(columns=cat).astype(float)
        X = pd.concat([num.fillna(num.median()),
                       pd.get_dummies(raw[cat], prefix={c: f"{c}_is" for c in cat}).astype(float)], axis=1)
    return traj, X


def _prettify(p):
    p = re.sub(r"_change_(\d+)_(\d+)", r" change \1-\2", p.strip())
    p = re.sub(r"_(\d{4})\b", r" in \1", p)
    p = re.sub(r"_is_", " = ", p).replace("_", " ")
    return re.sub(r"([\d.]+e[+\-]?\d+|\d+\.\d+)", lambda m: f"{float(m.group()):.2f}", p)


def _parse_rules(text):
    out = {}
    for seg in text.split("|"):
        seg = seg.strip()
        if seg.startswith("default"):
            continue
        m = re.match(r"G(\d+):\s*(.*)", seg)
        if not m:
            continue
        clauses = re.split(r"\)\s*OR\s*\(", m.group(2).strip())
        out[int(m.group(1))] = "\nor  ".join(
            ",  ".join(_prettify(x) for x in re.split(r"\s+AND\s+", c.strip().strip("()"))) for c in clauses)
    return out


def _num(x):
    a = abs(x)
    if a >= 10:
        return f"{round(x):d}"
    return f"{x:.1f}" if a >= 1 else f"{x:.2g}"


def _smooth(y, w):
    if w <= 1:
        return y
    k = np.ones(w) / w
    pad = np.r_[np.repeat(y[0], w // 2), y, np.repeat(y[-1], w - w // 2 - 1)]
    return np.convolve(pad, k, "valid")


def _short_rule(text, width=34, maxlines=2):
    """One or two compact lines: merge same-feature bounds into an interval
    (feature in (lo, hi]), round thresholds, keep only the first OR-branch."""
    if not text:
        return ""
    first = text.split("\nor")[0].strip()
    extra = text.count("\nor")
    bounds, singles = {}, []
    for p in (q.strip() for q in re.split(r",\s{1,}", first) if q.strip()):
        m = re.match(r"^(.*?)\s*(<=|>|=)\s*(-?[\d.]+(?:e[+\-]?\d+)?)$", p)
        if not m:
            singles.append(p); continue
        name, op, val = m.group(1).strip(), m.group(2), float(m.group(3))
        lo, hi = bounds.get(name, (None, None))
        bounds[name] = (val, hi) if op == ">" else (lo, val) if op == "<=" else (val, val)
    parts = []
    for name, (lo, hi) in bounds.items():
        if lo is not None and hi is not None and lo != hi:
            parts.append(f"{name} in ({_num(lo)}, {_num(hi)}]")
        elif lo is not None:
            parts.append(f"{name} > {_num(lo)}")
        elif hi is not None:
            parts.append(f"{name} ≤ {_num(hi)}")
        else:
            parts.append(name)
    s = ", ".join(parts + singles) + (f"  (+{extra} alt.)" if extra else "")
    lines = textwrap.wrap(s, width=width)
    if len(lines) > maxlines:
        lines = lines[:maxlines]
        lines[-1] = lines[-1].rstrip(", ") + "…"
    return "\n".join(lines)


def render(ds, traj, labels, rules_text):
    """Small multiples: one panel per subgroup, shared y-axis, population mean
    dashed, member spread as a light band (10th-90th percentile), subgroup mean
    bold, the short rule and the count in each title. Light smoothing for reading."""
    _, _, title, ylabel, xlabel = PANELS[ds]
    lab = pd.Series(labels, index=traj.index).astype(int)
    rules = _parse_rules(rules_text)
    sizes = lab.value_counts()
    no_rule = [g for g in sizes.index if g not in rules]
    default_id = no_rule[0] if no_rule else int(sizes.idxmax())
    nd = sorted([g for g in sizes.index if g != default_id])
    order = nd + [default_id]
    colmap = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(nd)}
    colmap[default_id] = "#6e6e6e"

    cols = traj.columns.to_numpy()
    xs = np.arange(len(cols))
    w = max(1, len(cols) // 16)
    pop = _smooth(traj.mean(axis=0).to_numpy(), w)

    n = len(order)
    ncol = min(3, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.7 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    axf = axes.ravel()
    for ax, g in zip(axf, order):
        col = colmap[g]
        mem = traj.loc[lab[lab == g].index].to_numpy()
        lo = _smooth(np.percentile(mem, 10, axis=0), w)
        hi = _smooth(np.percentile(mem, 90, axis=0), w)
        gm = _smooth(mem.mean(axis=0), w)
        ax.fill_between(xs, lo, hi, color=col, alpha=0.16, linewidth=0)
        ax.plot(xs, pop, "--", color="0.45", lw=1.3, zorder=3)
        ax.plot(xs, gm, color=col, lw=2.6, zorder=4)
        is_def = g == default_id
        head = "Default (no rule)" if is_def else _short_rule(rules.get(g, ""))
        ax.set_title(f"{head}\n[ n = {len(mem)} ]", fontsize=8.5,
                     color=col if not is_def else "0.3", loc="left", linespacing=1.35)
        ax.spines[["top", "right"]].set_visible(False)
        step = max(1, len(cols) // 5)
        ax.set_xticks(xs[::step])
        ax.set_xticklabels([str(cols[i]) for i in range(0, len(cols), step)])
    for ax in axf[n:]:
        ax.set_visible(False)
    for r in range(nrow):
        axes[r][0].set_ylabel(ylabel, fontsize=10)
    for c in range(ncol):
        axes[nrow - 1][c].set_xlabel(xlabel, fontsize=10)
    fig.suptitle(title, fontsize=12, fontweight="bold", x=0.01, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=2.4, w_pad=1.2)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{ds}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    datasets = sys.argv[1].split(",") if len(sys.argv) > 1 else list(PANELS)
    for ds in datasets:
        traj, X = load(ds)
        res = run_seal(traj, X, PARAMS)
        b = res["best"]
        d = OUT / ds
        d.mkdir(parents=True, exist_ok=True)
        pd.Series(b["labels"], index=traj.index, name="subgroup").rename_axis("entity")\
            .to_csv(d / "membership.csv")
        (d / "rules.txt").write_text(b["rules"], encoding="utf-8")
        render(ds, traj, b["labels"], b["rules"])
        print(f"[{ds}] n={len(traj)} T={traj.shape[1]} descriptors={X.shape[1]} "
              f"cost={b['cost']:.4f} groups={b['n_groups']} -> figures/{ds}.pdf")
