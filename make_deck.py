"""Build deck.pdf (8 slides) from results_summary.json + result files.
Run after run_pipeline.py:  .venv/bin/python make_deck.py
Requires matplotlib (in the venv). No network needed.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
S = json.load(open(ROOT / "results_summary.json"))
S1 = S["stage1"]
S2 = S["stage2"]
S3 = S["stage3"]
CV = S1["cv_metrics_oof_train_only"]

NAVY = "#0b2a5b"
BLUE = "#1f5fd0"
LIGHT = "#eaf1fb"
GREY = "#5a6472"


def slide(title, lines, footer="Atomiq — Qiskit Fall Fest 2026 at IEM | Umar Shareef (solo)"):
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.set_xlim(0, 13.33)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 6.9), 13.33, 0.6, boxstyle="square,pad=0",
                                fc=NAVY, ec=NAVY))
    ax.text(0.45, 7.2, title, color="white", fontsize=22, fontweight="bold", va="center")
    y = 6.35
    for block in lines:
        if block.startswith("## "):
            ax.text(0.45, y, block[3:], color=NAVY, fontsize=17, fontweight="bold", va="top")
            y -= 0.52
        elif block.startswith("- "):
            ax.text(0.7, y, "•  " + block[2:], color="#1c2733", fontsize=14.5, va="top", wrap=True)
            y -= 0.42
        elif block == "":
            y -= 0.25
        else:
            ax.text(0.45, y, block, color="#1c2733", fontsize=14.5, va="top")
            y -= 0.42
    ax.text(0.45, 0.28, footer, color=GREY, fontsize=11, va="center")
    return fig


def table_slide(title, headers, rows, footer_lines=None):
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.set_xlim(0, 13.33)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 6.9), 13.33, 0.6, boxstyle="square,pad=0", fc=NAVY, ec=NAVY))
    ax.text(0.45, 7.2, title, color="white", fontsize=22, fontweight="bold", va="center")
    col_w = [4.4, 1.9, 1.9, 1.9, 1.9][:len(headers)]
    x0 = 0.45
    y = 6.3
    xs = [x0]
    for w in col_w[:-1]:
        xs.append(xs[-1] + w)
    for xi, h in zip(xs, headers):
        ax.text(xi, y, h, color=NAVY, fontsize=14, fontweight="bold", va="top")
    y -= 0.45
    ax.plot([0.45, 12.88], [y + 0.12, y + 0.12], color=NAVY, lw=1.5)
    for r in rows:
        for xi, c in zip(xs, r):
            ax.text(xi, y, str(c), color="#1c2733", fontsize=13.5, va="top")
        y -= 0.45
    if footer_lines:
        fy = y - 0.35
        for line in footer_lines:
            ax.text(0.45, fy, line, color=GREY, fontsize=12.5, va="top")
            fy -= 0.34
    ax.text(0.45, 0.28, "Atomiq — Qiskit Fall Fest 2026 at IEM | Umar Shareef (solo)",
            color=GREY, fontsize=11, va="center")
    return fig


_SHORT = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "grad_boosting": "Gradient boosting",
    "quantum_kernel_svc": "Quantum kernel SVC",
    "hybrid_avg_logreg_qkernel": "Hybrid (logreg + Q-kernel)",
    "hybrid_avg_rf_qkernel": "Hybrid (RF + Q-kernel)",
}


def cv_row(name):
    m = CV[name]
    return [_SHORT[name], m["roc_auc"], m["pr_auc"], m["brier"], m["log_loss"]]


slides = []
slides.append(slide("A hybrid quantum-classical campaign", [
    "## Hybrid Quantum-Classical Term Deposit Campaign Pipeline",
    "",
    "-  Bank term-deposit campaign: predict who subscribes, pick 6 customers",
    "   to contact under budget + capacity limits, estimate the chance at",
    "   least 3 of them subscribe.",
    f"-  FINAL ANSWER — contact: {', '.join(S2['selected_ids'])}",
    f"-  P(at least 3 of 6 subscribe) — quantum {S3['quantum_estimate_P_ge3']:.4f}  |  "
    f"exact {S3['exact_analytical_P_ge3']:.4f}  |  Monte Carlo {S3['monte_carlo_P_ge3_200k']:.4f}",
    "",
    "-  Solo entry — Umar Shareef, UAF BSCS. Qiskit + Aer simulator, seed 42.",
    "-  All three stages are linked: Stages 2 and 3 use the Stage 1 probabilities.",
]))
slides.append(slide("The pipeline: three linked stages", [
    "## One connected system — errors carry forward by design",
    "",
    "-  STAGE 1 — PREDICT: 300 labelled customers train a hybrid classifier;",
    "   probabilities for all 100 test customers. Top 10 move on.",
    "-  STAGE 2 — SELECT: QAOA picks exactly 6 of the top 10, maximising",
    "   expected net value under budget (25) and capacity (22).",
    "-  STAGE 3 — SIMULATE: one qubit per selected customer, RY angle set",
    "   from its probability; count outcomes with at least 3 ones.",
    "",
    "-  Quantum vs classical is separated at every stage, and every result",
    "   is checked against an exact classical reference.",
]))
slides.append(table_slide("Stage 1 — hybrid classifier (5-fold CV, train only)",
    ["Model", "ROC-AUC", "PR-AUC", "Brier", "Log loss"],
    [cv_row("logistic_regression"), cv_row("random_forest"), cv_row("grad_boosting"),
     cv_row("quantum_kernel_svc"), cv_row("hybrid_avg_logreg_qkernel"),
     cv_row("hybrid_avg_rf_qkernel")],
    [f"Final submitted model: {_SHORT[S1['final_model']]} — chosen by the pre-registered OOF PR-AUC rule (see README).",
     "Base subscription rate in train: 11.7% (35/300).",
     f"Quantum kernel: ZZFeatureMap (4 qubits, reps 2) on {', '.join(S1['quantum_features'])} —",
     "features chosen by train-only mutual information. Preprocessing fit on train only."]))
slides.append(slide("Stage 2 — QAOA selection", [
    "## Exactly 6 of the top 10, under budget and capacity",
    "",
    f"-  Candidates: top 10 by Stage 1 probability — {', '.join(S1['top10_ids'])}",
    f"-  Selected (QAOA): {', '.join(S2['selected_ids'])}",
    f"-  Objective (expected net value): QAOA {S2['objective_qaoa']:.2f}  |  "
    f"brute-force optimum {S2['objective_brute_force']:.2f}  |  gap {S2['optimality_gap']:.4f}",
    f"-  Totals: contact cost {S2['total_cost']:.0f} / 25  •  capacity {S2['total_capacity']:.0f} / 22  •  exactly 6 selected",
    "",
    "-  QUBO with penalty terms for the three constraints; decoded from",
    "   measured bitstrings, feasibility checked on every sampled string.",
    "-  Brute force over all 210 combinations confirms the optimum.",
]))
slides.append(slide("Stage 3 — 6-qubit outcome simulation", [
    "## One qubit per customer, RY angle from its probability",
    "",
    f"-  Qubits: {'; '.join(S3['selected_ids'])} — qubit i carries customer i,",
    "   RY(2·arcsin(√p)) so P(|1⟩) = p. Customers assumed independent.",
    f"-  P(≥3 subscribe): quantum {S3['quantum_estimate_P_ge3']:.4f} (16,384 shots, Aer)",
    f"   exact Poisson-binomial {S3['exact_analytical_P_ge3']:.4f}  •  "
    f"Monte Carlo (200k) {S3['monte_carlo_P_ge3_200k']:.4f}",
    f"-  Quantum − exact error: {S3['error_quantum_minus_exact']:+.4f}  •  "
    f"noisy-Aer estimate {S3['noisy_estimate_P_ge3']:.4f}",
    f"-  Transpiled circuit depth {S3['transpiled_circuit_depth']} — 6 RY gates + 6 measurements, no entangling gates,",
]))
slides.append(slide("Error analysis — what can go wrong, and how big it is", [
    "## Honest limits, measured not asserted",
    "",
    "-  Stage 1 is the bottleneck: only 35 positive examples in 300 train",
    "   rows, so ranking (PR-AUC ~0.20–0.23) is modest — and every later",
    "   stage inherits those probabilities unchanged.",
    "-  Stage 2: QAOA is heuristic; the brute-force check bounds its gap.",
    "   An infeasible selection would earn zero — ours is feasible.",
    "-  Stage 3: finite-shot error (~±0.004 at 16,384 shots) plus readout /",
    "   depolarising noise in the noisy run; both reported next to exact.",
    "",
    "-  No quantum-advantage claim: Aer simulator, small problem, classical",
    "   references beat or match the quantum steps — the quantum parts are",
    "   demonstrated, validated, and kept separate from classical baselines.",
]))
slides.append(slide("Business impact — what the bank should do", [
    "## Contact these 6, expect about half the target to convert",
    "",
    f"-  Contact {', '.join(S2['selected_ids'])} — total cost {S2['total_cost']:.0f} of a 25 budget,",
    f"   capacity {S2['total_capacity']:.0f} of 22. Expected net value {S2['objective_qaoa']:.2f}.",
    f"-  Chance at least 3 subscribe: ~{S3['exact_analytical_P_ge3']*100:.0f}% — the campaign",
    "   target is a coin-flip, not a certainty; do not over-commit follow-up.",
    "-  Next: re-score weekly as new outcomes arrive, refit on train only,",
    "   and re-run the same pipeline — the selection step is the cheap part.",
    "-  Do not contact the other top-10 names: they cost more per expected",
    "   deposit than the six chosen once budget and capacity bind.",
]))
slides.append(slide("Reproducibility — run it yourself", [
    "## Everything is seeded, pinned, and runnable",
    "",
    "-  .venv/bin/python run_pipeline.py  — regenerates every result file",
    "   (stage1_predictions.csv, stage2_selection.csv/.json,",
    "   stage3_simulation.csv/.json) from resources/ in one command.",
    "-  Seeds: 42 for data split, model CV, QAOA sampling, Aer shots,",
    "   transpiler, and Monte Carlo. Versions pinned in requirements.txt.",
    "-  Backends: AerSimulator only (no IBM Quantum account, no hardware).",
    "   QAOA via Qiskit Optimization + Qiskit Algorithms; kernel via",
    "   Qiskit Machine Learning FidelityQuantumKernel.",
    "",
    "-  Repo: public GitHub, MIT licence, README is the technical report.",
]))

from matplotlib.backends.backend_pdf import PdfPages
with PdfPages(ROOT / "deck.pdf") as pdf:
    for fig in slides:
        pdf.savefig(fig)
        plt.close(fig)
print(f"WROTE deck.pdf ({len(slides)} slides)")
