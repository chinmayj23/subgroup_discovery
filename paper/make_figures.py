"""Generate the publication figures for the SEAL case-study paper.

All figures share one visual grammar so the reader learns it once:
  - the global mean trajectory is a dashed black line,
  - the leftover 'rest' group is grey,
  - discovered subgroups use a colourblind-safe palette and are numbered G1, G2, ...
    by size,
  - the sample size n is always printed,
  - axes carry the real target name and the real time unit.

Outputs vector PDFs into paper/figures/ (drop straight into Overleaf).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "seal_improved_runs"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "cmie_sd_run"))
sys.path.insert(0, str(RUNS))

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

# dataset -> (target label, time unit, pretty title)
META = {
    "co2": ("CO2 per capita (t)", "year", "CO2 emissions"),
    "covid": ("Deaths per million", "week (0 near early 2020)", "COVID mortality"),
    "worldbank": ("GDP per capita (1990 = 1)", "year", "GDP per capita growth"),
    "epa": ("PM2.5 (ug/m3)", "month index", "PM2.5 air quality"),
}


def _load(ds):
    traj = pd.read_parquet(RUNS / ds / "trajectories.parquet")
    mem = pd.read_csv(RUNS / ds / "membership.csv").set_index("entity")["subgroup"]
    lab = mem.reindex(traj.index).to_numpy()
    return traj, lab


def _plot_panel(ax, traj, lab, ylabel, xlabel, title):
    times = list(traj.columns)
    xs = np.arange(len(times))
    ax.plot(xs, traj.mean(axis=0).to_numpy(), "--", color="black", lw=1.8,
            label=f"all (n={len(traj)})", zorder=6)
    ci = 0
    for g in sorted(set(lab)):
        idx = lab == g
        m = traj.iloc[idx].mean(axis=0).to_numpy()
        if g == 0:
            ax.plot(xs, m, color="0.62", lw=1.3, label=f"rest (n={idx.sum()})")
        else:
            ax.plot(xs, m, color=PALETTE[ci % len(PALETTE)], lw=2.2,
                    label=f"G{g} (n={idx.sum()})")
            ci += 1
    n = len(times)
    step = max(1, n // 6)
    ax.set_xticks(xs[::step])
    ax.set_xticklabels([str(times[i]) for i in range(0, n, step)], fontsize=7, rotation=0)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=7, frameon=False, ncol=1, loc="best")


def fig_case_trajectories():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, ds in zip(axes.ravel(), ["co2", "covid", "worldbank", "epa"]):
        traj, lab = _load(ds)
        ylabel, xlabel, title = META[ds]
        _plot_panel(ax, traj, lab, ylabel, xlabel, title)
    fig.tight_layout()
    fig.savefig(FIG / "fig_case_trajectories.pdf")
    plt.close(fig)
    print("wrote fig_case_trajectories.pdf")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
    boxes = [
        (0.2, "Longitudinal panel\nentities x time\ntarget trajectory"),
        (2.2, "Cumulative\ntrajectory +\nEuclidean distance"),
        (4.2, "Randomised\nclustering\n(k-means /\nagglomerative)"),
        (6.4, "Descriptors X\nstatic +\nengineered\ntemporal"),
        (8.4, "Decision tree\non descriptors"),
        (10.4, "Interpretable\nrules +\nsubgroups"),
    ]
    w, h, y = 1.7, 1.7, 0.7
    for x, txt in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    linewidth=1.2, edgecolor="#333333", facecolor="#EEF3F8"))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.3)
    for x, _ in boxes[:-1]:
        ax.add_patch(FancyArrowPatch((x + w, y + h / 2), (x + 2.0, y + h / 2),
                                     arrowstyle="-|>", mutation_scale=13, color="#333333", lw=1.1))
    ax.text(6.0, 0.28,
            "search minimises  cost = within-group / global distance  +  lambda x rule complexity;"
            "  diverse rule sets kept by adjusted-Rand ranking",
            ha="center", va="center", fontsize=8, style="italic", color="#444444")
    fig.savefig(FIG / "fig_pipeline.pdf")
    plt.close(fig)
    print("wrote fig_pipeline.pdf")


def fig_temporal_features():
    """Illustrate the engineered temporal descriptors on one real income-like series."""
    rng = np.random.default_rng(3)
    t = np.arange(0, 60)
    base = 100 + 0.15 * t
    y = base + rng.normal(0, 3, len(t))
    y[24:33] -= np.linspace(6, 34, 9)   # a sustained shortfall episode
    y[33:40] += np.linspace(4, 26, 7)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(t, y, color="#0072B2", lw=1.8, label="covariate series")
    coef = np.polyfit(t, y, 1)
    ax.plot(t, np.polyval(coef, t), "--", color="#333333", lw=1.3,
            label=f"trend (slope {coef[0]:.2f})")
    mu = y.mean(); ax.axhline(mu, color="0.7", lw=0.9, ls=":")
    ax.axvspan(24, 32, color="#D55E00", alpha=0.14)
    ax.annotate("worst sustained shortfall\nseverity, length, recency",
                xy=(28, y[24:33].min()), xytext=(36, y.min() + 4), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="#D55E00"))
    ax.scatter([t[-1]], [y[-1]], color="#009E73", zorder=5)
    ax.annotate("recent level", xy=(t[-1], y[-1]), xytext=(t[-1] - 15, y[-1] + 10),
                fontsize=8.5, arrowprops=dict(arrowstyle="->", color="#009E73"))
    ax.set_xlabel("time"); ax.set_ylabel("covariate value")
    ax.set_title("Engineered temporal descriptors from one covariate series")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig_temporal_features.pdf"); plt.close(fig)
    print("wrote fig_temporal_features.pdf")


def fig_tree_example(ds="worldbank"):
    """Render the decision tree that defines the subgroups (surrogate refit on the
    published labels reproduces the exact partition)."""
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    import importlib
    import run_improved_seal as R
    importlib.reload(R)
    prep_rel, target_col, ylabel, xlabel = R.DATASETS[ds]
    prep = pd.read_csv(ROOT / "seal_repo" / "data" / prep_rel)
    traj = R.build_traj(ROOT / "seal_repo" / "results" / ds / "part4_target_trajectories.csv")
    X = R.build_X(prep, target_col, traj.index.to_numpy())
    common = X.index.intersection(traj.index)
    X = X.loc[common].sort_index()
    lab = pd.read_csv(RUNS / ds / "membership.csv").set_index("entity")["subgroup"].reindex(X.index).to_numpy()
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, random_state=0).fit(X.to_numpy(), lab)
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_tree(tree, feature_names=list(X.columns),
              class_names=["rest" if c == 0 else f"G{c}" for c in tree.classes_],
              filled=True, rounded=True, impurity=False, fontsize=7, ax=ax)
    ax.set_title(f"Decision tree defining the {ds} subgroups")
    fig.savefig(FIG / "fig_tree_example.pdf"); plt.close(fig)
    print("wrote fig_tree_example.pdf")


def fig_cmie_window():
    """CMIE COVID-centered window and the completeness that drives cohort size."""
    tl_path = ROOT / "cmie_sd_run" / "cmie_results_seal" / "prepared" / "target_long.csv"
    if not tl_path.exists():
        print("skip fig_cmie_window (no local CMIE target_long)")
        return
    tl = pd.read_csv(tl_path)
    pres = (tl.assign(o=1).pivot_table(index="entity", columns="time", values="o", fill_value=0) > 0)
    arr = pres.to_numpy(); T = arr.shape[1]
    cal = [str(pd.Period("2014-01", freq="M") + i) for i in range(T)]
    coverage = arr.mean(axis=0)  # fraction of households observed each month
    fig, ax = plt.subplots(figsize=(9, 4.2))
    xs = np.arange(T)
    ax.plot(xs, coverage * 100, color="#0072B2", lw=1.8)
    s, e = 30, 30 + 90        # Jun 2016 .. Nov 2023 (slots 30..119)
    ax.axvspan(s, e - 1, color="#009E73", alpha=0.15, label="fit window (Jun 2016 - Nov 2023)")
    covid = 74
    ax.axvline(covid, color="#D55E00", lw=1.3, ls="--", label="COVID onset (Mar 2020)")
    step = 12
    ax.set_xticks(xs[::step]); ax.set_xticklabels([cal[i] for i in range(0, T, step)], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("month"); ax.set_ylabel("households observed (%)")
    ax.set_title("CMIE panel coverage and the COVID-centered fit window")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "fig_cmie_window.pdf"); plt.close(fig)
    print("wrote fig_cmie_window.pdf")


def fig_cmie_subgroups():
    """CMIE subgroups reconstructed from the published best rules applied to the
    local cohort (sizes match the server run exactly: 8801 / 2389 / 3462). Mean
    health-burden trajectory per subgroup over the 90-month window, same grammar
    as the public panels."""
    import logging
    import run_seal as R
    cmie = ROOT / "cmie_sd_run"
    prep = cmie / "cmie_results_seal" / "prepared"
    wd = cmie / "cmie_results_seal"
    if not (prep / "target_long.csv").exists():
        print("skip fig_cmie_subgroups (no local CMIE data)")
        return
    import json
    tl = pd.read_csv(prep / "target_long.csv")
    desc = pd.read_parquet(prep / "descriptors.parquet")
    cfg = json.loads((cmie / "config.json").read_text())
    fit = cfg["fit"]
    lg = logging.getLogger("cmie"); lg.setLevel(logging.ERROR)
    m2t = R._calendar(wd / "raw", sorted(tl["time"].unique()), lg)
    win, cohort = R._window_cohort(tl, m2t, fit, lg)
    keep = set(R._select_households(list(cohort), 0, 0))
    # target matrix over the window
    sub = tl[tl["entity"].isin(keep) & tl["time"].isin(set(win))]
    traj = R._pivot_interp(sub, win)
    # apply the published best rules to label groups
    d = desc.reindex(traj.index)
    age = pd.to_numeric(d["MAX_AGE"], errors="coerce")
    borr = pd.to_numeric(d["BORRPROP_BORR_FOR_CONS_EXP"], errors="coerce").fillna(0)
    lab = pd.Series(0, index=traj.index)          # default
    lab[borr > 0.2029] = 2                          # G5 -> plotted as second subgroup
    lab[(age > 60.5) & (borr <= 0.2029)] = 1        # G2 -> first subgroup
    lab = lab.to_numpy()
    # calendar labels for the window slots
    t2m = {v: k for k, v in m2t.items()}
    months = [t2m.get(t, str(t)) for t in win]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs = np.arange(len(win))
    ax.plot(xs, traj.mean(axis=0).to_numpy(), "--", color="black", lw=1.8,
            label=f"all households (n={len(traj)})", zorder=6)
    names = {1: ("elderly, low consumption borrowing", PALETTE[1]),
             2: ("high consumption borrowing", PALETTE[0])}
    ax.plot(xs, traj.iloc[lab == 0].mean(axis=0).to_numpy(), color="0.62", lw=1.4,
            label=f"rest (n={(lab==0).sum()})")
    for g, (desc_txt, col) in names.items():
        ax.plot(xs, traj.iloc[lab == g].mean(axis=0).to_numpy(), color=col, lw=2.3,
                label=f"{desc_txt} (n={(lab==g).sum()})")
    covid = win.index(m2t["Mar 2020"]) if "Mar 2020" in m2t and m2t["Mar 2020"] in win else None
    if covid is not None:
        ax.axvline(covid, color="#D55E00", lw=1.1, ls=":", zorder=2)
        ax.text(covid + 1, ax.get_ylim()[1] * 0.96, "COVID onset", fontsize=7.5, color="#D55E00")
    step = max(1, len(win) // 7)
    ax.set_xticks(xs[::step])
    ax.set_xticklabels([months[i] for i in range(0, len(win), step)], rotation=45, ha="right", fontsize=7.5)
    ax.set_xlabel("month")
    ax.set_ylabel("health spending share of income")
    ax.set_title("CMIE household subgroups over the COVID-centered window")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig_cmie_subgroups.pdf"); plt.close(fig)
    print("wrote fig_cmie_subgroups.pdf")


def fig_cmie_rulecard():
    """CMIE figure with the subgroup trajectories on the left and the rules that
    define them, in the matching colors, on the right."""
    import logging
    import run_seal as R
    cmie = ROOT / "cmie_sd_run"
    prep = cmie / "cmie_results_seal" / "prepared"
    wd = cmie / "cmie_results_seal"
    if not (prep / "target_long.csv").exists():
        print("skip fig_cmie_rulecard (no local CMIE data)")
        return
    import json
    tl = pd.read_csv(prep / "target_long.csv")
    desc = pd.read_parquet(prep / "descriptors.parquet")
    cfg = json.loads((cmie / "config.json").read_text())
    fit = cfg["fit"]
    lg = logging.getLogger("cmie"); lg.setLevel(logging.ERROR)
    m2t = R._calendar(wd / "raw", sorted(tl["time"].unique()), lg)
    win, cohort = R._window_cohort(tl, m2t, fit, lg)
    keep = set(R._select_households(list(cohort), 0, 0))
    sub = tl[tl["entity"].isin(keep) & tl["time"].isin(set(win))]
    traj = R._pivot_interp(sub, win)
    d = desc.reindex(traj.index)
    age = pd.to_numeric(d["MAX_AGE"], errors="coerce")
    borr = pd.to_numeric(d["BORRPROP_BORR_FOR_CONS_EXP"], errors="coerce").fillna(0)
    lab = pd.Series(0, index=traj.index)
    lab[borr > 0.2029] = 2
    lab[(age > 60.5) & (borr <= 0.2029)] = 1
    lab = lab.to_numpy()
    t2m = {v: k for k, v in m2t.items()}
    months = [t2m.get(t, str(t)) for t in win]

    ELDER, BORROW, REST = PALETTE[1], PALETTE[0], "0.55"
    fig = plt.figure(figsize=(10.6, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.10)
    ax = fig.add_subplot(gs[0])
    xs = np.arange(len(win))
    ax.plot(xs, traj.mean(axis=0).to_numpy(), "--", color="black", lw=1.6, zorder=6)
    ax.plot(xs, traj.iloc[lab == 0].mean(axis=0).to_numpy(), color=REST, lw=1.4)
    ax.plot(xs, traj.iloc[lab == 1].mean(axis=0).to_numpy(), color=ELDER, lw=2.4)
    ax.plot(xs, traj.iloc[lab == 2].mean(axis=0).to_numpy(), color=BORROW, lw=2.4)
    # mark the COVID dip as an observation, not a rule
    covid = win.index(m2t["Mar 2020"]) if m2t.get("Mar 2020") in win else None
    if covid is not None:
        gm = traj.mean(axis=0).to_numpy()
        trough = covid + int(np.argmin(gm[covid:covid + 4]))
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.0018, hi)
        ax.axvspan(covid, covid + 2, color="0.6", alpha=0.12, zorder=1)
        ax.text(covid + 1, hi, "COVID\nlockdown", fontsize=8.8, color="0.4",
                ha="center", va="top", linespacing=0.9)
        ax.annotate("spending falls faster than income", xy=(trough, gm[trough]),
                    xytext=(trough + 1.5, lo - 0.0012), fontsize=8.8, color="0.3",
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color="0.45", lw=0.9,
                                    connectionstyle="arc3,rad=0.15"))
    step = max(1, len(win) // 6)
    ax.set_xticks(xs[::step]); ax.set_xticklabels([months[i] for i in range(0, len(win), step)],
                                                  rotation=45, ha="right", fontsize=7.5)
    ax.set_xlabel("month")
    ax.set_ylabel("health spending share of income")

    # right panel with the colored rules
    rx = fig.add_subplot(gs[1]); rx.axis("off"); rx.set_xlim(0, 1); rx.set_ylim(0, 1)
    # boolean rule only, with a faithful natural-language reading in brackets
    cards = [
        (ELDER,  "Rule 1",
         "MAX_AGE > 60  and  cons. borrowing <= 0.20",
         "(oldest member over 60, little consumption borrowing)",
         "n = 2,389"),
        (BORROW, "Rule 2",
         "not Rule 1  and  cons. borrowing > 0.20",
         "(borrows heavily for consumption)",
         "n = 3,462"),
        (REST,   "Default", None,
         "(all remaining households)",
         "n = 8,801"),
    ]
    y = 0.92
    for i, (col, tag, mono, gloss, note) in enumerate(cards):
        if i > 0:
            rx.plot([0.0, 0.95], [y + 0.06, y + 0.06], color="0.85", lw=0.8)
        rx.add_patch(plt.Rectangle((0.0, y - 0.028), 0.045, 0.045, color=col, transform=rx.transAxes))
        rx.text(0.08, y, tag, fontsize=10.5, fontweight="bold", color=col, va="center")
        yy = y - 0.07
        if mono:
            rx.text(0.08, yy, mono, fontsize=7.8, va="top", color="0.12", family="monospace")
            yy -= 0.05
        rx.text(0.08, yy, gloss, fontsize=8.4, va="top", color="0.4", style="italic")
        yy -= 0.055
        rx.text(0.08, yy, note, fontsize=8.6, va="top", color=col)
        y -= 0.28
    fig.savefig(FIG / "fig_cmie.pdf"); plt.close(fig)
    print("wrote fig_cmie.pdf")


def fig_country_rulecard(ds, spec, ylabel, xlabel, out_name):
    """Rule-card figure for a country panel: subgroup mean trajectories on the
    left, and on the right each subgroup's rule, its example countries, and its
    size. Reads the supervisor's runs in seal_repo/results/<ds>/."""
    R = ROOT / "seal_repo" / "results" / ds
    p8 = pd.read_csv(R / "part8_subgroup_trajectories.csv")
    p9 = pd.read_csv(R / "part9_trajectories_by_group.csv")
    sizes = p9.drop_duplicates("entity").groupby("group").size()
    vcol = [c for c in p9.columns if c.startswith("target")][0]
    gmean = p9.groupby("time")[vcol].mean()
    times = sorted(p8["time"].unique())
    xs = np.arange(len(times))
    tidx = {t: i for i, t in enumerate(times)}

    ncards = len(spec)
    fig = plt.figure(figsize=(11.2, 3.0 + 0.62 * ncards))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.25], wspace=0.10)
    ax = fig.add_subplot(gs[0])
    ax.plot([tidx[t] for t in gmean.index if t in tidx],
            [gmean[t] for t in gmean.index if t in tidx], "--", color="black", lw=1.5, zorder=6)
    for card in spec:
        sub = p8[p8["group"] == card["group"]].sort_values("time")
        ax.plot([tidx[t] for t in sub["time"]], sub["mean_target"].to_numpy(),
                color=card["color"], lw=2.2 if card["group"] != "default group" else 1.4)
    step = max(1, len(times) // 6)
    ax.set_xticks(xs[::step]); ax.set_xticklabels([str(times[i]) for i in range(0, len(times), step)],
                                                  rotation=45, ha="right", fontsize=7.5)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    rx = fig.add_subplot(gs[1]); rx.axis("off"); rx.set_xlim(0, 1); rx.set_ylim(0, 1)
    step_y = 1.0 / ncards
    for i, card in enumerate(spec):
        col = card["color"]
        top = 1.0 - i * step_y
        if i > 0:
            rx.plot([0.0, 1.0], [top, top], color="0.88", lw=0.8)
        rx.plot(0.015, top - 0.028, "s", color=col, markersize=9, clip_on=False)
        rx.text(0.06, top - 0.028, card["head"], fontsize=10, fontweight="bold", color=col, va="center")
        yy = top - 0.075
        if card.get("rule"):
            rx.text(0.06, yy, card["rule"], fontsize=7.8, va="top", color="0.4", style="italic")
            yy -= 0.038
        rx.text(0.06, yy, card["ctry"], fontsize=8.4, va="top", color="0.12")
        yy -= 0.036
        rx.text(0.06, yy, f"n = {int(sizes.get(card['group'], 0))}", fontsize=8.2, va="top", color=col)
    fig.savefig(FIG / out_name); plt.close(fig)
    print("wrote", out_name)


def fig_ours_rulecard(ds, spec, ylabel, xlabel, out_name):
    """Rule-card figure from OUR runs (seal_improved_runs): subgroup mean
    trajectories on the left, and each subgroup's rule, example countries, and
    size on the right."""
    tr = pd.read_parquet(RUNS / ds / "trajectories.parquet")
    mem = pd.read_csv(RUNS / ds / "membership.csv").set_index("entity")["subgroup"].reindex(tr.index)
    sizes = mem.value_counts()
    times = list(tr.columns)
    xs = np.arange(len(times))

    ncards = len(spec)
    fig = plt.figure(figsize=(11.2, 3.0 + 0.62 * ncards))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.25], wspace=0.10)
    ax = fig.add_subplot(gs[0])
    ax.plot(xs, tr.mean(axis=0).to_numpy(), "--", color="black", lw=1.5, zorder=6)
    for card in spec:
        idx = mem[mem == card["gid"]].index
        ax.plot(xs, tr.loc[idx].mean(axis=0).to_numpy(),
                color=card["color"], lw=2.2 if card["gid"] != 0 else 1.4)
    step = max(1, len(times) // 6)
    ax.set_xticks(xs[::step]); ax.set_xticklabels([str(times[i]) for i in range(0, len(times), step)],
                                                  rotation=45, ha="right", fontsize=7.5)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    rx = fig.add_subplot(gs[1]); rx.axis("off"); rx.set_xlim(0, 1); rx.set_ylim(0, 1)
    step_y = 1.0 / ncards
    rule_k = 0
    for i, card in enumerate(spec):
        col = card["color"]
        top = 1.0 - i * step_y
        if i > 0:
            rx.plot([0.0, 1.0], [top, top], color="0.88", lw=0.8)
        rx.plot(0.015, top - 0.028, "s", color=col, markersize=9, clip_on=False)
        if card["gid"] == 0:
            head, prefix = "Default", ""
        else:
            rule_k += 1
            head = f"Rule {rule_k}"
            prefix = "" if rule_k == 1 else (
                "not Rule 1,  " if rule_k == 2 else f"not Rules 1-{rule_k - 1},  ")
        rx.text(0.06, top - 0.028, head, fontsize=10, fontweight="bold", color=col, va="center")
        yy = top - 0.072
        if card.get("rule"):
            rx.text(0.06, yy, prefix + card["rule"], fontsize=7.0, va="top", color="0.12", family="monospace")
            yy -= 0.042
        rx.text(0.06, yy, card["ctry"], fontsize=8.4, va="top", color="0.12")
        yy -= 0.036
        rx.text(0.06, yy, f"n = {int(sizes.get(card['gid'], 0))}", fontsize=8.2, va="top", color=col)
    fig.savefig(FIG / out_name); plt.close(fig)
    print("wrote", out_name)


P = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
WB_OURS = [
    dict(gid=1, color=P[0], head="Catch-up economies",
         rule="birth rate <= 25  and  birth rate falling",
         ctry="China, India, South Korea, Bangladesh, Indonesia"),
    dict(gid=2, color=P[1], head="High fertility, stalled infrastructure",
         rule="birth rate > 25  and  electricity access flat",
         ctry="Nigeria, Eq. Guinea, Chad, Niger, Iraq"),
    dict(gid=0, color="0.55", head="All other countries", rule=None,
         ctry="United States, Germany, United Kingdom, France, Brazil"),
]
CO2_OURS = [
    dict(gid=4, color=P[1], head="Wealthy, gas and oil",
         rule="GDP/cap > 25k,  gas > 0.30,  coal < 0.19",
         ctry="Qatar, UAE, Saudi Arabia, Kuwait, United States, Canada"),
    dict(gid=3, color=P[0], head="Wealthy, coal reliant",
         rule="GDP/cap > 25k,  coal > 0.19",
         ctry="Germany, Russia, South Korea, Japan, Australia, Poland"),
    dict(gid=1, color=P[2], head="Wealthy, lower carbon",
         rule="GDP/cap > 25k,  coal < 0.19,  gas < 0.30",
         ctry="France, Luxembourg, Ireland, Denmark"),
    dict(gid=2, color=P[3], head="Middle income, large population",
         rule="GDP/cap 10-25k,  large population,  oil < 0.64",
         ctry="China, Indonesia, Mexico, South Africa, Iran"),
    dict(gid=0, color="0.55", head="Lower income", rule=None,
         ctry="Bangladesh, Nigeria, Ethiopia, Kenya"),
]
COVID_OURS = [
    dict(gid=3, color=P[1], head="Older, very large countries",
         rule="median age > 25,  pop > 11M,  density < 139",
         ctry="United States, Brazil, Russia, Argentina, Mexico"),
    dict(gid=2, color=P[0], head="Older, mid-size and sparse",
         rule="median age > 25,  pop 1.7-11M,  density < 139",
         ctry="Bulgaria, Hungary, Czechia, Austria"),
    dict(gid=1, color=P[2], head="Older, small countries",
         rule="median age > 25,  pop < 1.7M",
         ctry="San Marino, Montenegro, Trinidad, small island states"),
    dict(gid=4, color=P[3], head="Older, dense countries",
         rule="median age > 25,  density > 139",
         ctry="Belgium, Germany, Netherlands, United Kingdom, Italy"),
    dict(gid=0, color="0.55", head="Younger populations", rule=None,
         ctry="Nigeria, Bangladesh, Bolivia"),
]


P = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
WB_SPEC = [
    dict(group="group 1", color=P[0], head="Catch-up economies",
         rule="internet use in 2000 <= 2.3%  and  birth rate in 2010 <= 22",
         ctry="China, India, Bangladesh, Vietnam, Indonesia"),
    dict(group="group 2", color=P[1], head="High fertility, resource reliant",
         rule="birth rate in 2010 > 31  and  services trade in 2000 > 15%",
         ctry="Eq. Guinea, Chad, Iraq, Angola, DR Congo"),
    dict(group="default group", color="0.55", head="All other countries", rule=None,
         ctry="United States, Germany, Japan, South Korea, Brazil"),
]
COVID_SPEC = [
    dict(group="group 4", color=P[1], head="Older, central and eastern Europe",
         rule="richer and older, many hospital beds",
         ctry="Bulgaria, Hungary, Czechia, Poland, Germany"),
    dict(group="group 2", color=P[0], head="Older, western Europe",
         rule="richer and older, life expectancy over 81",
         ctry="Italy, United Kingdom, Belgium, Netherlands"),
    dict(group="group 1", color=P[2], head="Older, the Americas",
         rule="richer and older, fewer hospital beds",
         ctry="United States, Brazil, Chile"),
    dict(group="group 3", color=P[3], head="Younger, higher income",
         rule="richer but a younger population",
         ctry="Mexico, Argentina, Colombia, Peru"),
    dict(group="default group", color="0.55", head="Lower income",
         rule="younger, lower income countries",
         ctry="India, Bangladesh, Nigeria, Bolivia"),
]
CO2_SPEC = [
    dict(group="group 4", color=P[1], head="Wealthy oil and gas producers",
         rule="high income, little coal in the mix",
         ctry="Qatar, UAE, Saudi Arabia, Kuwait"),
    dict(group="group 3", color=P[0], head="Wealthy, coal reliant",
         rule="high income, coal-heavy mix",
         ctry="United States, Germany, Russia, Australia, Canada"),
    dict(group="group 1", color=P[2], head="Other high income",
         rule="high income with an oil-leaning mix",
         ctry="France, Japan, Ireland"),
    dict(group="group 2", color=P[3], head="Middle income, growing",
         rule="middle income with a growing population",
         ctry="China, Brazil, Mexico, Iran, South Africa"),
    dict(group="default group", color="0.55", head="Lower income",
         rule="low income, low per-capita emissions",
         ctry="India, Bangladesh, Nigeria"),
]


if __name__ == "__main__":
    fig_pipeline()
    fig_temporal_features()
    fig_case_trajectories()
    fig_tree_example()
    fig_cmie_window()
    fig_cmie_subgroups()
    print("figures ->", FIG)
