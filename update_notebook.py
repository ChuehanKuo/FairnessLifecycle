"""
Update the Jupyter notebook to match the edited manuscript figure numbering.

Manuscript figure mapping (edited → current notebook):
  Fig 1  = Method heatmap (was Fig 5, Cell 74)
  Fig 2  = Deployability cliff (was Fig 8, Cell 83)
  Fig 3  = Lifecycle funnel (was Fig 6, Cell 77)
  Fig 4  = NEW: Benchmark-deployment mismatch (Panel A: fold gaps, Panel B: fairness vs AUROC)
  Fig 5  = Clinical impact (was Fig 7, Cell 81)

Supplementary figures:
  S1 Fig = Pareto frontier (Cell 56)
  S2 Fig = ROC curves by quintile (Cell 35)
  S3 Fig = SHAP importance (Cell 37)
  S4 Fig = Predicted probability distributions (Cell 36)
  S5 Fig = Fairea trade-off (was Fig 4, Cell 72)
  S6 Fig = NEW: Detailed lifecycle survival funnel (with method names)

Old main figures demoted to supplementary:
  Old Fig 2 (Cells 67-68): TPR gap ranked → supplementary
  Old Fig 3 (Cell 70): Category box plot → supplementary
"""

import json
import copy

NB_PATH = "SHARE_Fairness_Benchmark_v12__2___1_ (1).ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)

cells = nb["cells"]


def replace_source(cell_idx, old, new):
    """Replace text in a cell's source."""
    src = "".join(cells[cell_idx]["source"])
    if old not in src:
        print(f"  WARNING: '{old[:60]}' not found in cell {cell_idx}")
        return False
    src = src.replace(old, new)
    cells[cell_idx]["source"] = [src]
    return True


def set_source(cell_idx, new_src):
    """Set a cell's entire source."""
    cells[cell_idx]["source"] = [new_src]


def make_code_cell(source):
    """Create a new code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [source],
    }


def make_md_cell(source):
    """Create a new markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [source],
    }


# ============================================================
# 1. RENUMBER MAIN FIGURES
# ============================================================
print("=== Renumbering main figures ===")

# --- Cell 73 (md): Fig 5 header → Fig 1 ---
set_source(73, "### Fig 1: Method Heatmap \u2014 Consistency Across Outcomes\nShows percentage reduction in TPR gap for all methods \u00d7 all outcomes.  \nReveals: the same top methods work everywhere; augmentation fails everywhere.")
print("  Cell 73 md: Fig 5 → Fig 1")

# --- Cell 74 (code): Fig 5 heatmap → Fig 1 ---
replace_source(74, "# FIG 5: Method Heatmap", "# FIG 1: Method Heatmap")
replace_source(74, "Fig5_method_heatmap.png", "Fig1_method_heatmap.png")
replace_source(74, "'✓ Fig5_method_heatmap.png'", "'✓ Fig1_method_heatmap.png'")
# Also fix the print statement variant
src74 = "".join(cells[74]["source"])
src74 = src74.replace("Fig5_method_heatmap", "Fig1_method_heatmap")
cells[74]["source"] = [src74]
print("  Cell 74 code: Fig5 → Fig1")

# --- Cell 82 (md): Fig 8 header → Fig 2 ---
set_source(82, "### Fig 2: The Deployability Cliff\nProportion of deployable methods by category. Highlights that **0/6 post-processing** methods are deployable.")
print("  Cell 82 md: Fig 8 → Fig 2")

# --- Cell 83 (code): Fig 8 deployability cliff → Fig 2 ---
src83 = "".join(cells[83]["source"])
src83 = src83.replace("FIG 8", "FIG 2")
src83 = src83.replace("deployability_cliff.png", "Fig2_deployability_cliff.png")
src83 = src83.replace("deployability_cliff.pdf", "Fig2_deployability_cliff.pdf")
cells[83]["source"] = [src83]
print("  Cell 83 code: Fig8 → Fig2")

# --- Cell 76 (md): Fig 6 header → Fig 3 ---
set_source(76, "### Fig 3: Lifecycle Survival Funnel\nShows progressive elimination of 26 methods through deployment stages.\nThis is the paper\u2019s **signature figure**.")
print("  Cell 76 md: Fig 6 → Fig 3")

# --- Cell 77 (code): Fig 6 lifecycle funnel → Fig 3 ---
src77 = "".join(cells[77]["source"])
src77 = src77.replace("FIG 6", "FIG 3")
src77 = src77.replace("lifecycle_funnel.png", "Fig3_lifecycle_funnel.png")
src77 = src77.replace("lifecycle_funnel.pdf", "Fig3_lifecycle_funnel.pdf")
cells[77]["source"] = [src77]
print("  Cell 77 code: Fig6 → Fig3")

# --- Cell 80 (md): Fig 7 header → Fig 5 ---
set_source(80, "### Fig 5: Clinical Impact Estimation\nTranslates TPR gap reductions into additional correct identifications per screening wave.")
print("  Cell 80 md: Fig 7 → Fig 5")

# --- Cell 81 (code): Fig 7 clinical impact → Fig 5 ---
src81 = "".join(cells[81]["source"])
src81 = src81.replace("FIG 7", "FIG 5")
src81 = src81.replace("clinical_impact.png", "Fig5_clinical_impact.png")
src81 = src81.replace("clinical_impact.pdf", "Fig5_clinical_impact.pdf")
cells[81]["source"] = [src81]
print("  Cell 81 code: Fig7 → Fig5")


# ============================================================
# 2. DEMOTE OLD MAIN FIGURES TO SUPPLEMENTARY
# ============================================================
print("\n=== Demoting old main figures to supplementary ===")

# --- Cell 66 (md): Old Fig 2 header → Supplementary ---
set_source(66, "### Supplementary: TPR Gap Ranked by Method (with performance metrics)\nHorizontal bar chart ranked by TPR gap. Table columns show AUROC, Sensitivity, Balanced Accuracy, Specificity.  \nDashed red line = baseline. Methods above the line improved fairness.  \n*Note: This figure is no longer a main manuscript figure; retained as supplementary material.*")
print("  Cell 66 md: Fig 2 → Supplementary")

# --- Cells 67-68 (code): Old Fig 2 → Supplementary ---
src67 = "".join(cells[67]["source"])
src67 = src67.replace("# FIG 2: TPR Gap Ranked Bar Charts", "# SUPPLEMENTARY: TPR Gap Ranked Bar Charts (formerly Fig 2)")
src67 = src67.replace("TPRgap_", "Supp_TPRgap_")
cells[67]["source"] = [src67]

src68 = "".join(cells[68]["source"])
src68 = src68.replace("# FIG 2 COMPOSITE", "# SUPPLEMENTARY: TPR Gap Composite (formerly Fig 2)")
src68 = src68.replace("Fig2_TPRgap_composite.png", "Supp_TPRgap_composite.png")
cells[68]["source"] = [src68]
print("  Cells 67-68 code: Fig2 → Supplementary")

# --- Cell 69 (md): Old Fig 3 header → Supplementary ---
set_source(69, "### Supplementary: TPR Gap by Method Category\nBox plots showing that method category predicts effectiveness.  \nPost-processing and in-processing sit below baseline; augmentation does not.  \n*Note: This figure is no longer a main manuscript figure; retained as supplementary material.*")
print("  Cell 69 md: Fig 3 → Supplementary")

# --- Cell 70 (code): Old Fig 3 → Supplementary ---
src70 = "".join(cells[70]["source"])
src70 = src70.replace("# FIG 3: Category Box Plot", "# SUPPLEMENTARY: Category Box Plot (formerly Fig 3)")
src70 = src70.replace("Fig3_category_boxplot.png", "Supp_category_boxplot.png")
cells[70]["source"] = [src70]
print("  Cell 70 code: Fig3 → Supplementary")

# --- Cell 71 (md): Old Fig 4 header → S5 Fig ---
set_source(71, "### S5 Fig: Fairea Trade-off Classification\nScatter plot: TPR gap reduction (x) vs AUROC change (y).  \nUpper-right = effective fairness at low cost. Colour = category.  \n*Supplementary Figure S5 in the manuscript.*")
print("  Cell 71 md: Fig 4 → S5 Fig")

# --- Cell 72 (code): Old Fig 4 Fairea → S5 Fig ---
src72 = "".join(cells[72]["source"])
src72 = src72.replace("# FIG 4: Fairea Trade-off Classification", "# S5 FIG: Fairea Trade-off Classification (formerly Fig 4)")
src72 = src72.replace("Fig4_fairea_tradeoff.png", "S5_fairea_tradeoff.png")
cells[72]["source"] = [src72]
print("  Cell 72 code: Fig4 → S5 Fig")


# ============================================================
# 3. RELABEL SUPPLEMENTARY FIGURE CELLS
# ============================================================
print("\n=== Relabeling supplementary figure cells ===")

# --- Cell 35 (code): ROC curves → S2 Fig ---
src35 = "".join(cells[35]["source"])
src35 = src35.replace(">>> SUPPLEMENT S3: ROC Curves", ">>> S2 Fig: ROC Curves by Income Quintile")
src35 = src35.replace("roc_subgroup_combined.png", "S2_roc_by_quintile.png")
cells[35]["source"] = [src35]
print("  Cell 35 code: → S2 Fig")

# --- Cell 36 (code): KDE by SES → S4 Fig ---
src36 = "".join(cells[36]["source"])
src36 = src36.replace("kde_by_ses_combined.png", "S4_predicted_probabilities.png")
cells[36]["source"] = [src36]
print("  Cell 36 code: → S4 Fig")

# --- Cell 37 (code): SHAP → S3 Fig ---
src37 = "".join(cells[37]["source"])
src37 = src37.replace(">>> SUPPLEMENT S4: SHAP Plots", ">>> S3 Fig: SHAP Feature Importance")
src37 = src37.replace("shap_combined.png", "S3_shap_importance.png")
cells[37]["source"] = [src37]
print("  Cell 37 code: → S3 Fig")

# --- Cell 55 (md): Pareto caption → S1 Fig ---
set_source(55, "### S1 Fig: Pareto Frontier\nPareto frontier of performance (AUROC) vs fairness (TPR gap) by track. Star markers indicate Pareto-optimal methods \u2014 no other method in the same track simultaneously achieves better performance and fairness. Dashed line connects the frontier; green shading indicates the dominated region.\n*Supplementary Figure S1 in the manuscript.*")
print("  Cell 55 md: → S1 Fig")

# --- Cell 56 (code): Pareto → S1 Fig ---
src56 = "".join(cells[56]["source"])
src56 = src56.replace(">>> FIGURE 5 (Paper): Pareto Frontier", ">>> S1 Fig: Pareto Frontier per Outcome")
src56 = src56.replace("pareto_{out}_2track.png", "S1_pareto_{out}.png")
# Also update the print statement
src56 = src56.replace("pareto_2track.png", "S1_pareto.png")
src56 = src56.replace("\u2713 Figure 5 saved (2-panel per outcome)", "\u2713 S1 Fig saved (2-panel per outcome)")
cells[56]["source"] = [src56]
print("  Cell 56 code: → S1 Fig")

# --- Cell 57 (code): Fold stability → keep as S3 Table support ---
src57 = "".join(cells[57]["source"])
src57 = src57.replace(">>> SUPPLEMENT S6: Fold Stability", ">>> Fold Stability Plot (supports S3 Table and Fig 4A)")
src57 = src57.replace("fold_stability_combined.png", "Supp_fold_stability.png")
cells[57]["source"] = [src57]
print("  Cell 57 code: → supports S3 Table / Fig 4A")


# ============================================================
# 4. UPDATE SECTION HEADER
# ============================================================
print("\n=== Updating section header ===")
set_source(65, """---
## 20. Publication Figures

**Manuscript figure mapping (updated to match edited manuscript):**

| Figure | Description | Output File |
|--------|-------------|-------------|
| Fig 1 | Method heatmap (\u2014 % TPR gap reduction) | `Fig1_method_heatmap.png` |
| Fig 2 | Deployability cliff by category | `Fig2_deployability_cliff.png` |
| Fig 3 | Lifecycle survival funnel | `Fig3_lifecycle_funnel.png` |
| Fig 4 | Benchmark-deployment mismatch (A: fold gaps, B: fairness vs AUROC) | `Fig4_benchmark_mismatch.png` |
| Fig 5 | Projected clinical impact | `Fig5_clinical_impact.png` |
| S1 Fig | Pareto frontier per outcome | `S1_pareto_{outcome}.png` |
| S2 Fig | ROC curves by income quintile | `S2_roc_by_quintile.png` |
| S3 Fig | SHAP feature importance | `S3_shap_importance.png` |
| S4 Fig | Predicted probability distributions | `S4_predicted_probabilities.png` |
| S5 Fig | Fairea trade-off classification | `S5_fairea_tradeoff.png` |
| S6 Fig | Lifecycle survival funnel (detailed) | `S6_lifecycle_funnel_detailed.png` |""")
print("  Cell 65 md: Updated section header with figure mapping")


# ============================================================
# 5. INSERT NEW Fig 4: Benchmark-Deployment Mismatch
# ============================================================
print("\n=== Inserting new Fig 4 ===")

fig4_md = make_md_cell("""### Fig 4: Benchmark-Deployment Mismatch
Two-panel figure quantifying the gap between benchmark performance and real-world deployability.
- **Panel A**: Fold-level TPR gap variability \u2014 post-processing methods show high instability across cross-validation folds.
- **Panel B**: Fairness (\u0394TPR gap) vs. discriminative performance (AUROC) with deployability status overlaid.

*This is a NEW figure created for the edited manuscript.*""")

fig4_code = make_code_cell(r"""# ============================================================
# FIG 4 \u2013 Benchmark-Deployment Mismatch (2-panel)
# Panel A: Fold-level TPR gap variability (deployable vs non-deployable)
# Panel B: Fairness vs AUROC scatter with deployability overlay
# ============================================================
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

OUT = Path("fairness_output")
OUT.mkdir(exist_ok=True)

full = pd.read_csv("tables/full_results.csv")
deploy_df = pd.read_csv("tables/deployability_contract.csv")
per_fold = pd.read_csv("tables/per_fold_results.csv")
baselines = full[full["Method"] == "BASELINE"].set_index("Outcome")
methods_df = full[full["Method"] != "BASELINE"].copy()

tier1 = set(deploy_df[deploy_df["Deployable"] == "Yes"]["Method"])
outcomes = ["EUROD", "LS", "CASP", "SRH"]

OUTCOME_LABELS = {
    "EUROD": "Depression (EURO-D)",
    "LS": "Life Satisfaction",
    "CASP": "Quality of Life (CASP-12)",
    "SRH": "Self-Rated Health",
}

CATEGORY_COLORS = {
    "Pre-processing": "#4BACC6",
    "In-processing": "#6AAF4E",
    "Post-processing": "#E04040",
    "Augmentation": "#4472C4",
}

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor("white")

# ── Panel A: Fold-level TPR gap variability ──
# Show per-fold TPR gaps for each method, grouped by deployability
# Average across outcomes for clarity
method_fold_data = []
for method in sorted(methods_df["Method"].unique()):
    is_deploy = method in tier1
    cat = methods_df[methods_df["Method"] == method].iloc[0]["Category"]

    fold_gaps = []
    for oc in outcomes:
        pf = per_fold[(per_fold["Method"] == method) & (per_fold["Outcome"] == oc)]
        if not pf.empty:
            fold_gaps.extend(pf["TPR_gap"].values)

    if fold_gaps:
        mean_gap = np.nanmean(fold_gaps)
        sd_gap = np.nanstd(fold_gaps, ddof=1)
        method_fold_data.append({
            "Method": method, "mean": mean_gap, "sd": sd_gap,
            "deployable": is_deploy, "category": cat,
            "fold_gaps": fold_gaps
        })

# Sort by deployability then by mean gap
method_fold_data.sort(key=lambda x: (not x["deployable"], x["mean"]))

y_pos = np.arange(len(method_fold_data))
for i, md in enumerate(method_fold_data):
    color = "#2E8B57" if md["deployable"] else "#CC3333"
    alpha = 1.0 if md["deployable"] else 0.6

    # Individual fold dots
    ax_a.scatter(md["fold_gaps"], [i] * len(md["fold_gaps"]),
                 color=color, s=12, alpha=0.35, zorder=3)
    # Mean + SD bar
    ax_a.plot([md["mean"] - md["sd"], md["mean"] + md["sd"]], [i, i],
              color=color, linewidth=2, alpha=alpha)
    ax_a.plot(md["mean"], i, "D", color=color, markersize=5, zorder=5,
              markeredgecolor="black", markeredgewidth=0.5)

# Add baseline reference lines per outcome (averaged)
bl_mean = np.mean([baselines.loc[oc, "TPR_gap_mean"] for oc in outcomes])
ax_a.axvline(x=bl_mean, color="gray", linestyle=":", linewidth=1, alpha=0.6)
ax_a.text(bl_mean + 0.003, len(method_fold_data) - 0.5, f"Baseline\n({bl_mean:.3f})",
          fontsize=7, color="gray", va="top")

# Separator line between deployable and non-deployable
n_deploy = sum(1 for md in method_fold_data if md["deployable"])
if 0 < n_deploy < len(method_fold_data):
    ax_a.axhline(y=n_deploy - 0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax_a.text(ax_a.get_xlim()[1] * 0.95, n_deploy - 0.7, "Deployable \u2191",
              fontsize=8, color="#2E8B57", ha="right", fontweight="bold")
    ax_a.text(ax_a.get_xlim()[1] * 0.95, n_deploy - 0.3, "Non-deployable \u2193",
              fontsize=8, color="#CC3333", ha="right", fontweight="bold")

ax_a.set_yticks(y_pos)
ax_a.set_yticklabels([md["Method"] for md in method_fold_data], fontsize=7)
ax_a.set_xlabel("TPR Gap (per fold, all outcomes)", fontsize=10)
ax_a.set_title("A. Fold-Level TPR Gap Variability", fontsize=12, fontweight="bold")
ax_a.invert_yaxis()
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)
ax_a.grid(True, alpha=0.2, axis="x")

# ── Panel B: Fairness vs AUROC with deployability ──
# For each method: mean across outcomes of (% TPR gap reduction) vs (mean AUROC)
for _, row in methods_df.iterrows():
    method = row["Method"]
    oc = row["Outcome"]
    bl_auroc = baselines.loc[oc, "AUROC_mean"]
    bl_gap = baselines.loc[oc, "TPR_gap_mean"]

    is_deploy = method in tier1
    cat = row["Category"]
    marker = "o" if is_deploy else "X"
    edge = "black" if is_deploy else "none"
    size = 70 if is_deploy else 50
    alpha = 0.85 if is_deploy else 0.45
    color = CATEGORY_COLORS.get(cat, "#999999")

    ax_b.scatter(row["Pct_Reduction"], row["AUROC_mean"],
                 c=color, marker=marker, s=size, alpha=alpha,
                 edgecolors=edge, linewidths=0.5, zorder=3 if is_deploy else 2)

# Reference lines
ax_b.axhline(y=np.mean([baselines.loc[oc, "AUROC_mean"] for oc in outcomes]),
             color="gray", linestyle=":", linewidth=1, alpha=0.5)
ax_b.axvline(x=0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

# Annotate survivors
survivors = ["ExpGrad_TPR", "ExpGrad_EO", "Reweighing"]
for surv in survivors:
    sdata = methods_df[methods_df["Method"] == surv]
    for _, sr in sdata.iterrows():
        ax_b.annotate(surv, (sr["Pct_Reduction"], sr["AUROC_mean"]),
                      fontsize=6, fontweight="bold", color="#333",
                      xytext=(5, 5), textcoords="offset points",
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                edgecolor="#2E8B57", alpha=0.8))

ax_b.set_xlabel("% Reduction in TPR Gap (\u2192 fairer)", fontsize=10)
ax_b.set_ylabel("AUROC (\u2191 more accurate)", fontsize=10)
ax_b.set_title("B. Fairness vs. Performance by Deployability", fontsize=12, fontweight="bold")
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)
ax_b.grid(True, alpha=0.15)

# Legend
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
           markeredgecolor="black", markersize=8, label="Deployable"),
    Line2D([0], [0], marker="X", color="w", markerfacecolor="gray",
           markersize=8, label="Non-deployable"),
    mpatches.Patch(facecolor="#4BACC6", label="Pre-processing"),
    mpatches.Patch(facecolor="#6AAF4E", label="In-processing"),
    mpatches.Patch(facecolor="#E04040", label="Post-processing"),
    mpatches.Patch(facecolor="#4472C4", label="Augmentation"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=6, fontsize=9,
           frameon=True, bbox_to_anchor=(0.5, -0.02), edgecolor="#CCCCCC")

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(OUT / "Fig4_benchmark_mismatch.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT / "Fig4_benchmark_mismatch.pdf", bbox_inches="tight")
plt.show()
print(f"Saved: {OUT}/Fig4_benchmark_mismatch.png")
""")

# Insert after Cell 83 (Fig 2 deployability cliff) - at the end of main figures
# Actually, let's insert after the lifecycle summary (Cell 85) but before download (Cell 86)
# Position: after Cell 83 to keep it near other main figures
# Insert in reverse order so indices don't shift
insert_pos = 84  # After cell 83 (Fig 2)
cells.insert(insert_pos, fig4_code)
cells.insert(insert_pos, fig4_md)
print(f"  Inserted Fig 4 markdown + code at position {insert_pos}")


# ============================================================
# 6. INSERT NEW S6 Fig: Detailed Lifecycle Survival Funnel
# ============================================================
print("\n=== Inserting S6 Fig (detailed lifecycle funnel) ===")

s6_md = make_md_cell("""### S6 Fig: Lifecycle Survival Funnel (Detailed)
Detailed version of Fig 3 showing **method names** at each lifecycle stage.
Each stage bar lists the methods that survived to that point.
*Supplementary Figure S6 in the manuscript.*""")

s6_code = make_code_cell(r"""# ============================================================
# S6 FIG \u2013 Lifecycle Survival Funnel (Detailed with method names)
# ============================================================
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path("fairness_output")
OUT.mkdir(exist_ok=True)

full = pd.read_csv("tables/full_results.csv")
deploy = pd.read_csv("tables/deployability_contract.csv")

baselines = full[full["Method"] == "BASELINE"].set_index("Outcome")
methods = full[full["Method"] != "BASELINE"].copy()
outcomes = ["EUROD", "LS", "CASP", "SRH"]
all_methods = sorted(methods["Method"].unique())
n_start = len(all_methods)

CATEGORY_COLORS = {
    "Pre-processing": "#4BACC6",
    "In-processing": "#6AAF4E",
    "Post-processing": "#E04040",
    "Augmentation": "#4472C4",
}

# ── Apply lifecycle filters (same logic as Fig 3) ──
def check_effectiveness(m):
    rows = methods[methods["Method"] == m]
    return sum(rows["Pct_Reduction"] > 0) >= 3

stage1 = [m for m in all_methods if check_effectiveness(m)]

tier1 = set(deploy[deploy["Deployable"] == "Yes"]["Method"])
stage2 = [m for m in stage1 if m in tier1]

def check_accuracy(m):
    for oc in outcomes:
        bl = baselines.loc[oc, "AUROC_mean"]
        row = methods[(methods["Method"] == m) & (methods["Outcome"] == oc)]
        if row.empty or bl - row.iloc[0]["AUROC_mean"] > 0.03:
            return False
    return True
stage3 = [m for m in stage2 if check_accuracy(m)]

def check_stability(m):
    for oc in outcomes:
        row = methods[(methods["Method"] == m) & (methods["Outcome"] == oc)]
        if row.empty or row.iloc[0]["Sign_improved"] < 4:
            return False
    return True
stage4 = [m for m in stage3 if check_stability(m)]

def check_fairness(m):
    for oc in outcomes:
        row = methods[(methods["Method"] == m) & (methods["Outcome"] == oc)]
        if row.empty or row.iloc[0]["Pct_Reduction"] < 25:
            return False
    return True
stage5 = [m for m in stage4 if check_fairness(m)]

stages = [
    ("All 26 methods", all_methods),
    ("Stage 1: Effectiveness\n(reduces disparity in \u22653/4 outcomes)", stage1),
    ("Stage 2: Deployability\n(no protected attribute at inference)", stage2),
    ("Stage 3: Accuracy\n(AUROC drop \u22640.03)", stage3),
    ("Stage 4: Stability\n(sign \u22654/5 folds)", stage4),
    ("Stage 5: Fairness\n(\u226525% reduction all outcomes)", stage5),
]

# ── Plot detailed funnel ──
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor("white")

max_w = 0.85
y_positions = list(range(len(stages) - 1, -1, -1))
colors = ["#4A90D9", "#5BA0E0", "#7CB5E8", "#A8CCF0", "#C4DCF5", "#2E8B57"]

cat_map = methods.drop_duplicates("Method").set_index("Method")["Category"].to_dict()

for i, (label, method_list) in enumerate(stages):
    n = len(method_list)
    w = max_w * (n / len(all_methods)) if len(all_methods) > 0 else 0.1
    w = max(w, 0.15)

    rect = mpatches.FancyBboxPatch(
        (0.5 - w/2, y_positions[i] - 0.4), w, 0.8,
        boxstyle="round,pad=0.05", facecolor=colors[i], edgecolor="white", linewidth=2
    )
    ax.add_patch(rect)

    # Count in center
    ax.text(0.5, y_positions[i] + 0.15, f"{n}",
            ha="center", va="center", fontsize=20, fontweight="bold", color="white")

    # Stage label on the left
    ax.text(-0.08, y_positions[i], label.replace("\n", " "),
            ha="right", va="center", fontsize=9, color="#333")

    # Method names on the right, color-coded by category
    if n <= 15:
        method_str_parts = []
        for m in sorted(method_list):
            cat = cat_map.get(m, "Unknown")
            method_str_parts.append(m)
        method_text = ", ".join(sorted(method_list))
        ax.text(0.5 + w/2 + 0.03, y_positions[i], method_text,
                ha="left", va="center", fontsize=6.5, color="#555",
                fontstyle="italic", wrap=True)
    else:
        # Too many to list - show categories
        cats_count = {}
        for m in method_list:
            cat = cat_map.get(m, "Unknown")
            cats_count[cat] = cats_count.get(cat, 0) + 1
        cat_text = ", ".join([f"{cat}: {cnt}" for cat, cnt in sorted(cats_count.items())])
        ax.text(0.5 + w/2 + 0.03, y_positions[i], f"({cat_text})",
                ha="left", va="center", fontsize=7, color="#777")

    # Elimination annotation
    if i < len(stages) - 1:
        eliminated = len(stages[i][1]) - len(stages[i+1][1])
        if eliminated > 0:
            lost = set(stages[i][1]) - set(stages[i+1][1])
            ax.annotate(f"\u2212{eliminated}", xy=(0.5, y_positions[i] - 0.48),
                       fontsize=9, ha="center", color="#CC3333", fontweight="bold")
            if len(lost) <= 8:
                lost_text = ", ".join(sorted(lost))
                ax.text(0.5, y_positions[i] - 0.65, f"Lost: {lost_text}",
                       ha="center", fontsize=5.5, color="#CC3333", fontstyle="italic")

ax.set_xlim(-0.7, 1.6)
ax.set_ylim(-1.2, len(stages) - 0.1)
ax.set_title("S6 Fig: Detailed Lifecycle Survival Funnel\nFrom 26 Methods to 3 Survivors",
             fontsize=14, fontweight="bold", pad=15)
ax.axis("off")

# Category legend
legend_els = [mpatches.Patch(facecolor=c, label=cat, edgecolor="black")
              for cat, c in CATEGORY_COLORS.items()]
fig.legend(handles=legend_els, loc="lower center", ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig(OUT / "S6_lifecycle_funnel_detailed.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT / "S6_lifecycle_funnel_detailed.pdf", bbox_inches="tight")
plt.show()
print(f"Saved: {OUT}/S6_lifecycle_funnel_detailed.png")
""")

# Insert after the Fig 4 cells we just added (which are at positions 84-85)
# After our Fig 4 insert, the download cell moved to 88.
# Insert S6 before the download cell, after the lifecycle summary stats.
# The lifecycle summary was at cell 85 (now shifted to 87 because we inserted 2 cells).
insert_pos2 = 88  # After lifecycle summary (originally cell 85, now 87)
cells.insert(insert_pos2, s6_code)
cells.insert(insert_pos2, s6_md)
print(f"  Inserted S6 Fig markdown + code at position {insert_pos2}")


# ============================================================
# 7. SAVE NOTEBOOK
# ============================================================
print("\n=== Saving notebook ===")
with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Saved: {NB_PATH}")
print(f"Total cells: {len(cells)}")
print("\nDone! All figures renumbered and new cells inserted.")
