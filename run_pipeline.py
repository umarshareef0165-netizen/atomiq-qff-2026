"""
Atomiq — Qiskit Fall Fest 2026 (IEM): Hybrid Quantum-Classical Term Deposit Campaign Pipeline.
End-to-end, reproducible, seed-fixed (SEED = 42 everywhere).

Stage 1 — Predict subscription probability for 100 test customers.
           Classical baselines (logistic regression, random forest, gradient
           boosting) + a genuine quantum kernel classifier (ZZFeatureMap +
           fidelity kernel + SVC). Final model chosen by honest out-of-fold
           PR-AUC on TRAIN ONLY (test never used for selection/tuning).
Stage 2 — Select exactly 6 of the top 10 by QAOA (binary optimisation with
           budget/capacity/cardinality constraints), checked against exact
           brute-force optimum.
Stage 3 — 6-qubit RY circuit estimating P(at least 3 of the 6 subscribe),
           validated against the exact Poisson-binomial result and Monte Carlo.

Outputs (repo root): stage1_predictions.csv, stage2_selection.csv/.json,
stage3_simulation.csv/.json, results_summary.json
Run:  .venv/bin/python run_pipeline.py
"""
import itertools
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

SEED = 42
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "resources" / "bank_campaign_participant_release.xlsx"
SHOTS_STAGE3 = 16384
SHOTS_QAOA = 4096

rng = np.random.default_rng(SEED)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

NUMERIC = ["age", "balance", "day", "campaign", "pdays", "previous"]
CATEGORICAL = ["job", "marital", "education", "default", "housing",
               "loan", "contact", "month", "poutcome"]

summary = {"seed": SEED, "shots_stage3": SHOTS_STAGE3, "shots_qaoa": SHOTS_QAOA}


def cv_metrics(y_true, oof_proba):
    return {
        "roc_auc": round(float(roc_auc_score(y_true, oof_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, oof_proba)), 4),
        "brier": round(float(brier_score_loss(y_true, oof_proba)), 4),
        "log_loss": round(float(log_loss(y_true, oof_proba)), 4),
    }


# ---------------------------------------------------------------- load data
train = pd.read_excel(DATA, sheet_name="Stage1_Train")
test = pd.read_excel(DATA, sheet_name="Stage1_Test")
stage2_data = pd.read_excel(DATA, sheet_name="Stage2_Data")
assert len(train) == 300 and len(test) == 100, (len(train), len(test))
assert set(test["customer_id"]).issubset(set(stage2_data["customer_id"]))
assert set(train["customer_id"]).isdisjoint(set(test["customer_id"]))
y = (train["y"].str.lower() == "yes").astype(int).to_numpy()
print(f"train={len(train)} test={len(test)} train positives={y.sum()} ({y.mean():.3f})")

X_train_raw, X_test_raw = train[NUMERIC + CATEGORICAL], test[NUMERIC + CATEGORICAL]

# ------------------------------------------------------- Stage 1: classical
preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])
classical_models = {
    "logistic_regression": Pipeline([("prep", preprocess),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED))]),
    "random_forest": Pipeline([("prep", preprocess),
        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                       random_state=SEED, n_jobs=-1))]),
    "grad_boosting": Pipeline([("prep", preprocess),
        ("clf", HistGradientBoostingClassifier(random_state=SEED))]),
}
oof = {}
stage1_cv = {}
for name, model in classical_models.items():
    proba = cross_val_predict(model, X_train_raw, y, cv=skf, method="predict_proba")[:, 1]
    oof[name] = proba
    stage1_cv[name] = cv_metrics(y, proba)
    print(f"CV {name}: {stage1_cv[name]}")

# ------------------------------------------------- Stage 1: quantum kernel
# Train-only feature choice for the 4-qubit kernel: mutual information on
# standardised numerics (fit on train only — test never touched).
scaler_num = StandardScaler().fit(train[NUMERIC])
Xn = scaler_num.transform(train[NUMERIC])
mi = mutual_info_classif(Xn, y, random_state=SEED)
q_features = [NUMERIC[i] for i in np.argsort(mi)[::-1][:4]]
print("quantum-kernel features (train-only MI):",
      [(f, round(float(mi[NUMERIC.index(f)]), 4)) for f in q_features])

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel

scaler_q = StandardScaler().fit(train[q_features])  # fit on train only
Xq_train = scaler_q.transform(train[q_features])
Xq_test = scaler_q.transform(test[q_features])
feature_map = zz_feature_map(feature_dimension=4, reps=2, entanglement="linear")
# Fidelity kernel K_ij = |<phi(x_i)|phi(x_j)>|^2, computed exactly from
# statevectors (one 16-dim matmul — mathematically identical to the
# FidelityQuantumKernel sampler evaluation, but minutes faster). The
# qiskit-machine-learning kernel is still run on a 6x6 subset as a
# cross-check that the two agree.
from qiskit.quantum_info import Statevector


def _states(X):
    return np.array([Statevector(feature_map.assign_parameters(list(row))).data
                     for row in X])


_cache = ROOT / "_kernel_cache.npz"
if _cache.exists():
    _c = np.load(_cache)
    if list(_c["q_features"]) == q_features:
        K_train, K_test = _c["K_train"], _c["K_test"]
        print("loaded cached quantum kernels")
    else:
        _cache.unlink()
        K_train = K_test = None
else:
    K_train = K_test = None
if K_train is None:
    Sx = _states(Xq_train)
    Sy = _states(Xq_test)
    K_train = np.abs(Sx @ Sx.conj().T) ** 2
    K_test = np.abs(Sy @ Sx.conj().T) ** 2
    np.savez(_cache, K_train=K_train, K_test=K_test, q_features=np.array(q_features))
    _chk = FidelityQuantumKernel(feature_map=feature_map)
    _ref = np.asarray(_chk.evaluate(x_vec=Xq_train[:6]))
    _diff = float(np.max(np.abs(_ref - K_train[:6, :6])))
    print(f"kernel cross-check vs FidelityQuantumKernel (6x6): max diff {_diff:.2e}")
    assert _diff < 1e-6, "manual fidelity kernel disagrees with qml kernel"
assert K_train.shape == (300, 300) and K_test.shape == (100, 300)

# 5-fold CV on the precomputed kernel (slicing only — kernel itself uses
# features only, no labels, so slicing is leakage-free).
oof_q = np.zeros(len(y))
for tr_idx, va_idx in skf.split(K_train, y):
    svc = SVC(kernel="precomputed", C=1.0, class_weight="balanced",
              probability=True, random_state=SEED)
    svc.fit(K_train[np.ix_(tr_idx, tr_idx)], y[tr_idx])
    oof_q[va_idx] = svc.predict_proba(K_train[np.ix_(va_idx, tr_idx)])[:, 1]
oof["quantum_kernel_svc"] = oof_q
stage1_cv["quantum_kernel_svc"] = cv_metrics(y, oof_q)
print(f"CV quantum_kernel_svc: {stage1_cv['quantum_kernel_svc']}")

# Hybrids: plain averages of a classical model and the quantum kernel
# classifier (genuine hybrid probabilistic classifiers).
oof["hybrid_avg_logreg_qkernel"] = 0.5 * oof["logistic_regression"] + 0.5 * oof_q
oof["hybrid_avg_rf_qkernel"] = 0.5 * oof["random_forest"] + 0.5 * oof_q
for _n in ["hybrid_avg_logreg_qkernel", "hybrid_avg_rf_qkernel"]:
    stage1_cv[_n] = cv_metrics(y, oof[_n])
    print(f"CV {_n}: {stage1_cv[_n]}")

# ---- final-model decision (pre-registered rule, train OOF only) ----
# Best out-of-fold PR-AUC among all six candidates wins; ties broken by
# Brier score. Every candidate — including the quantum kernel classifier
# and both hybrids — is built, validated and reported either way, so the
# quantum component is never hidden, even if a classical model validates
# better on this small training set.
_cands = ["logistic_regression", "random_forest", "grad_boosting",
          "quantum_kernel_svc", "hybrid_avg_logreg_qkernel", "hybrid_avg_rf_qkernel"]
final_model = min(_cands, key=lambda n: (-stage1_cv[n]["pr_auc"], stage1_cv[n]["brier"]))
print(f"FINAL Stage 1 model: {final_model} (best OOF PR-AUC, Brier tie-break)")

# Fit finals on the full training set (preprocessing fit on train only).
final_logreg = classical_models["logistic_regression"].fit(X_train_raw, y)
p_logreg_test = final_logreg.predict_proba(X_test_raw)[:, 1]
final_rf = classical_models["random_forest"].fit(X_train_raw, y)
p_rf_test = final_rf.predict_proba(X_test_raw)[:, 1]
svc_full = SVC(kernel="precomputed", C=1.0, class_weight="balanced",
               probability=True, random_state=SEED).fit(K_train, y)
p_q_test = svc_full.predict_proba(K_test)[:, 1]
_p = {"logistic_regression": p_logreg_test, "random_forest": p_rf_test,
      "quantum_kernel_svc": p_q_test,
      "hybrid_avg_logreg_qkernel": 0.5 * p_logreg_test + 0.5 * p_q_test,
      "hybrid_avg_rf_qkernel": 0.5 * p_rf_test + 0.5 * p_q_test}
if final_model == "grad_boosting":
    _gb = classical_models["grad_boosting"].fit(X_train_raw, y)
    p_test = _gb.predict_proba(X_test_raw)[:, 1]
else:
    p_test = _p[final_model]
p_test = np.clip(p_test, 0.001, 0.999)

s1 = pd.DataFrame({"customer_id": test["customer_id"], "predicted_probability": np.round(p_test, 6)})
s1 = s1.sort_values(["predicted_probability", "customer_id"],
                    ascending=[False, True]).reset_index(drop=True)
s1["rank"] = np.arange(1, len(s1) + 1)
s1["is_top_10"] = (s1["rank"] <= 10).astype(int)
s1 = s1[["customer_id", "predicted_probability", "rank", "is_top_10"]]
s1.to_csv(ROOT / "stage1_predictions.csv", index=False)
print("Stage 1 top 10:")
print(s1.head(10).to_string(index=False))
summary["stage1"] = {
    "cv_metrics_oof_train_only": stage1_cv,
    "quantum_features": q_features,
    "final_model": final_model,
    "final_model_rule": "best out-of-fold PR-AUC among logreg / random forest / grad boosting / quantum-kernel SVC / two hybrids; Brier tie-break",
    "top10_ids": s1.head(10)["customer_id"].tolist(),
    "test_probability_range": [round(float(p_test.min()), 4), round(float(p_test.max()), 4)],
}

# ---------------------------------------------------------------- Stage 2
top10 = s1.head(10).merge(stage2_data, on="customer_id", how="left")
assert len(top10) == 10 and top10[["contact_cost", "expected_deposit_value",
                                   "capacity_units"]].notna().all().all()
top10["expected_net_value"] = (top10["predicted_probability"]
                               * top10["expected_deposit_value"] - top10["contact_cost"])
top10 = top10.reset_index(drop=True)
ids10 = top10["customer_id"].tolist()
v = top10["expected_net_value"].to_numpy(float)
cost = top10["contact_cost"].to_numpy(float)
cap = top10["capacity_units"].to_numpy(float)
print("Stage 2 candidates (net value = p*deposit - cost):")
print(top10[["customer_id", "predicted_probability", "expected_deposit_value",
             "contact_cost", "capacity_units", "expected_net_value"]].to_string(index=False))

# Exact classical optimum by brute force over C(10,6)=210 combinations.
best = None
for combo in itertools.combinations(range(10), 6):
    c, h, z = cost[list(combo)].sum(), cap[list(combo)].sum(), v[list(combo)].sum()
    if c <= 25 and h <= 22 and (best is None or z > best["objective"]):
        best = {"combo": combo, "objective": float(z), "cost": float(c), "capacity": float(h)}
print(f"Brute-force optimum: {[ids10[i] for i in best['combo']]} Z={best['objective']:.4f} "
      f"cost={best['cost']} cap={best['capacity']}")

# QAOA via Qiskit Optimization (penalty-converted QUBO, p tried 2 then 3).
from qiskit import QuantumCircuit as _QC, transpile
from qiskit.primitives import StatevectorSampler
from qiskit_aer import AerSimulator as _AerSimulator
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo

qp = QuadraticProgram("atomiq_stage2")
for i in range(10):
    qp.binary_var(name=f"x{i}")
qp.minimize(linear=-v)  # maximise sum(v_i x_i)
qp.linear_constraint(linear=np.ones(10), sense="==", rhs=6, name="exactly_six")
qp.linear_constraint(linear=cost, sense="<=", rhs=25, name="budget")
qp.linear_constraint(linear=cap, sense="<=", rhs=22, name="capacity")
# --- QUBO construction (manual, 10 qubits, no slack variables) ---
# The two inequality constraints are VERIFIED vacuous for this data: over
# all 210 six-subsets, the largest possible total contact cost is 16 (< 25)
# and the largest possible total capacity is 18 (< 22) — so every 6-subset
# satisfies them automatically. The only binding constraint is SUM(x) = 6,
# enforced with a quadratic penalty P * (SUM(x) - 6)^2, P = 15 (a one-unit
# violation costs 15, more than the largest single net value, 6.92, so no
# infeasible bitstring can beat the best feasible one: the thinnest margin
# is a 7-subset at 15 + 33.22 - 40.14 = 8.08). Feasibility (cost/cap) is
# still checked on every sampled bitstring below. Building the QUBO by hand
# avoids the converter's slack variables, which would inflate the problem
# to 20 qubits for constraints that never bind.
_max_cost_6 = max(cost[list(c)].sum() for c in itertools.combinations(range(10), 6))
_max_cap_6 = max(cap[list(c)].sum() for c in itertools.combinations(range(10), 6))
assert _max_cost_6 <= 25 and _max_cap_6 <= 22, (_max_cost_6, _max_cap_6)
print(f"inequalities verified vacuous: max 6-subset cost={_max_cost_6} (<=25), "
      f"max capacity={_max_cap_6} (<=22)")
_PEN = 15.0
qubo = QuadraticProgram("atomiq_stage2_qubo")
for i in range(10):
    qubo.binary_var(name=f"x{i}")
_lin = {i: float(-v[i] - 11 * _PEN) for i in range(10)}  # -v_i + P*(1 - 12)
_quad = {(i, j): float(2 * _PEN) for i in range(10) for j in range(i + 1, 10)}
qubo.minimize(linear=_lin, quadratic=_quad, constant=36 * _PEN)

# --- Manual QAOA (exact statevector expectation) ---
# Same QAOA ansatz and variational loop as the library QAOA, but the cost
# expectation is evaluated exactly from the statevector against the
# precomputed diagonal QUBO energies — the Sampler-based library loop is
# minutes slower per evaluation on this VM and computes the same number.
# QAOAAnsatz parameter order is [beta_0..beta_{p-1}, gamma_0..gamma_{p-1}].
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector as _Statevector
from scipy.optimize import minimize as _minimize

ising_op, _ising_offset = qubo.to_ising()


def _build_qaoa_ansatz(ising_op, p):
    """QAOA ansatz built from plain RZZ/RZ/RX gates — exactly the same
    unitary as QAOAAnsatz (all ZZ/Z cost terms commute, so the product of
    individual rotations is exact, no Trotter error), but Statevector
    evaluation is ~100x faster than decomposing PauliEvolutionGate terms
    on every optimiser iteration. Layer order per rep: cost, then mixer.
    Parameters are [gamma_0..gamma_{p-1}, beta_0..beta_{p-1}]."""
    n = ising_op.num_qubits
    zz_terms, z_terms = [], []
    for pauli, coeff in zip(ising_op.paulis, ising_op.coeffs):
        c = float(np.real(coeff))
        if abs(c) < 1e-12:
            continue
        s = str(pauli)  # qubit 0 is the rightmost character
        zs = [n - 1 - k for k, ch in enumerate(s) if ch == "Z"]
        if len(zs) == 2:
            zz_terms.append((zs[0], zs[1], c))
        elif len(zs) == 1:
            z_terms.append((zs[0], c))
        elif len(zs) == 0:
            pass  # identity term = constant offset, no gate
        else:
            raise ValueError(f"non-diagonal Ising term: {s}")
    gamma = ParameterVector("gamma", p)
    beta = ParameterVector("beta", p)
    qc = _QC(n)
    qc.h(range(n))
    for layer in range(p):
        for (i, j, c) in zz_terms:
            qc.rzz(2 * gamma[layer] * c, i, j)
        for (i, c) in z_terms:
            qc.rz(2 * gamma[layer] * c, i)
        for i in range(n):
            qc.rx(2 * beta[layer], i)
    return qc
# Diagonal QUBO energies for all 1024 bitstrings (bit i of the integer = x_i).
_energies = np.array([float(qubo.objective.evaluate(
    [int((s >> i) & 1) for i in range(10)])) for s in range(2 ** 10)])


def _qaoa_expectation(theta, ansatz):
    sv = _Statevector(ansatz.assign_parameters(list(theta)))
    probs = np.abs(sv.data) ** 2
    return float(probs @ _energies)


qaoa_runs = []
qaoa_best = None
for p in (2, 3):
    ansatz = _build_qaoa_ansatz(ising_op, p)
    best_theta, best_energy = None, float("inf")
    _ramp_g = [0.3 + 0.6 * l / max(p - 1, 1) for l in range(p)]
    _ramp_b = [0.9 - 0.6 * l / max(p - 1, 1) for l in range(p)]
    _inits = ([0.5] * p + [0.5] * p, [0.25] * p + [1.0] * p,
              _ramp_g + _ramp_b, [1.0] * p + [0.25] * p)
    for _init in _inits:
        _res = _minimize(lambda th: _qaoa_expectation(th, ansatz),
                         x0=np.array(_init, dtype=float), method="COBYLA",
                         options={"maxiter": 400, "rhobeg": 0.5})
        if _res.fun < best_energy:
            best_energy, best_theta = float(_res.fun), _res.x
    # Sample the optimised circuit on Aer for the measured-bitstring
    # distribution (seed-fixed). Aer bitstrings are little-endian:
    # the rightmost character is qubit 0, i.e. bit i from the right = x_i.
    bound = ansatz.assign_parameters(list(best_theta))
    meas = _QC(10, 10)
    meas.compose(bound, inplace=True)
    meas.measure(range(10), range(10))
    sim_qaoa = _AerSimulator(seed_simulator=SEED)
    counts = sim_qaoa.run(transpile(meas, sim_qaoa, seed_transpiler=SEED),
                          shots=SHOTS_QAOA).result().get_counts()
    # Also the exact most-probable bitstring from the statevector, and the
    # probability the optimised state puts on the brute-force optimum.
    _sv = _Statevector(bound)
    _sv_probs = np.abs(_sv.data) ** 2
    _top_state = int(np.argmax(_sv_probs))
    _top_bits = [int((_top_state >> i) & 1) for i in range(10)]
    _opt_state = sum(1 << i for i in best["combo"])
    _prob_optimum = float(_sv_probs[_opt_state])
    sampled = []
    for bitstr, cnt in sorted(counts.items(), key=lambda kv: -kv[1])[:8]:
        bits = [int(bitstr[::-1][i]) for i in range(10)]  # index i -> x_i
        sel = [i for i, b in enumerate(bits) if b == 1]
        csum, hsum, zsum = cost[sel].sum(), cap[sel].sum(), v[sel].sum()
        feasible = len(sel) == 6 and csum <= 25 and hsum <= 22
        sampled.append({"bitstring": bitstr, "count": int(cnt),
                        "selected": [ids10[i] for i in sel], "objective": round(float(zsum), 4),
                        "cost": float(csum), "capacity": float(hsum), "feasible": bool(feasible)})
    _x = np.array(_top_bits, dtype=int)
    _sel = [i for i, b in enumerate(_top_bits) if b == 1]
    run = {"qaoa_reps": p, "optimizer": "scipy COBYLA(maxiter=200, 2 fixed inits), "
                                        "exact statevector expectation",
           "optimal_theta_gamma_then_beta": [round(float(t), 6) for t in best_theta],
           "optimal_expectation_qubo_energy": round(best_energy, 4),
           "probability_of_brute_force_optimum": round(_prob_optimum, 4),
           "most_probable_bitstring_x": [int(b) for b in _x],
           "most_probable_selected": [ids10[i] for i in _sel],
           "most_probable_objective": round(float(v[_sel].sum()), 4) if _sel else None,
           "most_probable_cost": float(cost[_sel].sum()) if _sel else None,
           "most_probable_capacity": float(cap[_sel].sum()) if _sel else None,
           "most_probable_feasible": bool(len(_sel) == 6 and cost[_sel].sum() <= 25
                                          and cap[_sel].sum() <= 22),
           "top_sampled_bitstrings": sampled}
    qaoa_runs.append(run)
    _cands = [s for s in sampled if s["feasible"]]
    if run["most_probable_feasible"]:
        _cands.append({"bitstring": "most-probable-statevector", "count": None,
                       "selected": run["most_probable_selected"],
                       "objective": run["most_probable_objective"],
                       "cost": run["most_probable_cost"],
                       "capacity": run["most_probable_capacity"], "feasible": True})
    if _cands:
        cand = max(_cands, key=lambda s: s["objective"])
        if qaoa_best is None or cand["objective"] > qaoa_best["objective"]:
            qaoa_best = cand
    print(f"QAOA p={p}: most-probable x={run['most_probable_selected']} "
          f"feasible={run['most_probable_feasible']} Z={run['most_probable_objective']} "
          f"(expectation {best_energy:.4f})")
    if qaoa_best is not None and abs(qaoa_best["objective"] - best["objective"]) < 1e-9:
        print(f"QAOA found the brute-force optimum at p={p}; stopping.")
        break

assert qaoa_best is not None, "QAOA sampled no feasible bitstring"
selected_ids = qaoa_best["selected"]
sel_idx = [ids10.index(c) for c in selected_ids]
sel_rows = top10.iloc[sel_idx]
# Gap from full-precision net values (not the 4-dp rounded display values),
# so an exact match reports exactly 0.0 rather than -0.0.
gap = round(best["objective"] - float(v[sel_idx].sum()), 4)
print(f"QAOA best feasible: {qaoa_best['selected']} Z={qaoa_best['objective']} gap={gap}")

s2_json = {
    "top10_candidates": [
        {"customer_id": r["customer_id"], "predicted_probability": round(float(r["predicted_probability"]), 6),
         "expected_deposit_value": int(r["expected_deposit_value"]), "contact_cost": int(r["contact_cost"]),
         "capacity_units": int(r["capacity_units"]),
         "expected_net_value": round(float(r["expected_net_value"]), 4)}
        for _, r in top10.iterrows()],
    "selected_customer_ids": selected_ids,
    "total_contact_cost": float(cost[sel_idx].sum()),
    "total_capacity_units": float(cap[sel_idx].sum()),
    "budget_limit": 25, "capacity_limit": 22, "exactly_k": 6,
    "objective_value_qaoa": qaoa_best["objective"],
    "objective_value_brute_force_optimum": round(best["objective"], 4),
    "optimality_gap": gap,
    "brute_force_optimum_ids": [ids10[i] for i in best["combo"]],
    "qaoa_runs": qaoa_runs,
}
with open(ROOT / "stage2_selection.json", "w") as f:
    json.dump(s2_json, f, indent=2)
s2_csv = top10[["customer_id", "predicted_probability", "expected_deposit_value",
                "contact_cost", "capacity_units", "expected_net_value"]].copy()
s2_csv["expected_net_value"] = s2_csv["expected_net_value"].round(4)
s2_csv["selected"] = [1 if cid in selected_ids else 0 for cid in s2_csv["customer_id"]]
s2_csv.to_csv(ROOT / "stage2_selection.csv", index=False)
summary["stage2"] = {
    "selected_ids": selected_ids,
    "objective_qaoa": qaoa_best["objective"],
    "objective_brute_force": round(best["objective"], 4),
    "optimality_gap": gap,
    "total_cost": float(cost[sel_idx].sum()),
    "total_capacity": float(cap[sel_idx].sum()),
}

# ---------------------------------------------------------------- Stage 3
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

probs = np.array([float(top10.loc[top10["customer_id"] == cid, "predicted_probability"].iloc[0])
                  for cid in selected_ids])
angles = [float(2 * math.asin(math.sqrt(p))) for p in probs]  # RY angle, P(|1>) = p
print("Stage 3 probabilities:", [round(float(p), 4) for p in probs])
print("Stage 3 RY angles (rad):", [round(a, 4) for a in angles])


def build_stage3_circuit():
    qc = QuantumCircuit(6, 6)
    for i, th in enumerate(angles):
        qc.ry(th, i)
    qc.measure(range(6), range(6))
    return qc


qc = build_stage3_circuit()
sim = AerSimulator(seed_simulator=SEED)
tqc = transpile(qc, sim, seed_transpiler=SEED)
counts = sim.run(tqc, shots=SHOTS_STAGE3).result().get_counts()
n_success = sum(c for b, c in counts.items() if b.count("1") >= 3)
q_est = n_success / SHOTS_STAGE3

# Exact Poisson-binomial P(>=3) by dynamic programming.
dp = [1.0]
for p in probs:
    ndp = [0.0] * (len(dp) + 1)
    for k, val in enumerate(dp):
        ndp[k] += val * (1 - p)
        ndp[k + 1] += val * p
    dp = ndp
exact = float(sum(dp[3:]))
mc = float((rng.binomial(1, probs, size=(200_000, 6)).sum(axis=1) >= 3).mean())

# Noisy Aer run (same depolarising model as the starter noise demo).
noise = NoiseModel()
noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ["ry"])
noise.add_all_qubit_quantum_error(depolarizing_error(0.05, 2), ["cx"])
noise.add_all_qubit_readout_error(ReadoutError([[0.97, 0.03], [0.03, 0.97]]))
noisy_counts = sim.run(tqc, shots=SHOTS_STAGE3, noise_model=noise).result().get_counts()
noisy_est = sum(c for b, c in noisy_counts.items() if b.count("1") >= 3) / SHOTS_STAGE3

# Circuit stats (original + transpiled).
ops = tqc.count_ops()
s3 = {
    "selected_customer_ids": selected_ids,
    "qubit_order": [f"qubit {i} = {cid}" for i, cid in enumerate(selected_ids)],
    "bit_ordering_note": "Qiskit little-endian: the rightmost character of a measured "
                         "bitstring is qubit 0. Success counts bitstrings with >=3 ones, "
                         "which is order-independent; the stated qubit order is used for "
                         "the RY angles and the statevector check.",
    "input_probabilities": [round(float(p), 6) for p in probs],
    "ry_angles_rad": [round(a, 6) for a in angles],
    "backend": "AerSimulator (noiseless + noisy runs)",
    "shots": SHOTS_STAGE3,
    "measurement_counts_top10": dict(sorted(counts.items(), key=lambda kv: -kv[1])[:10]),
    "quantum_estimate_P_ge3": round(float(q_est), 6),
    "exact_analytical_P_ge3": round(exact, 6),
    "monte_carlo_P_ge3_200k": round(mc, 6),
    "error_quantum_minus_exact": round(float(q_est - exact), 6),
    "error_monte_carlo_minus_exact": round(float(mc - exact), 6),
    "noisy_estimate_P_ge3": round(float(noisy_est), 6),
    "error_noisy_minus_exact": round(float(noisy_est - exact), 6),
    "logical_qubits": 6, "physical_qubits": 6,
    "original_circuit_depth": int(qc.depth()),
    "transpiled_circuit_depth": int(tqc.depth()),
    "transpiled_gate_counts": {k: int(v) for k, v in ops.items()},
    "swap_count": int(ops.get("swap", 0)),
}
with open(ROOT / "stage3_simulation.json", "w") as f:
    json.dump(s3, f, indent=2)
pd.DataFrame([{
    "customer_id": cid, "qubit": i, "predicted_probability": round(float(p), 6),
    "ry_angle_rad": round(float(a), 6)} for i, (cid, p, a) in
    enumerate(zip(selected_ids, probs, angles))]).to_csv(
        ROOT / "stage3_simulation.csv", index=False)
print(f"Stage 3: quantum={q_est:.4f} exact={exact:.4f} MC={mc:.4f} noisy={noisy_est:.4f}")
print(f"  transpiled depth={tqc.depth()} gates={dict(ops)}")

summary["stage3"] = {k: s3[k] for k in
                     ["quantum_estimate_P_ge3", "exact_analytical_P_ge3",
                      "monte_carlo_P_ge3_200k", "error_quantum_minus_exact",
                      "noisy_estimate_P_ge3", "transpiled_circuit_depth",
                      "transpiled_gate_counts"]}
summary["stage3"]["selected_ids"] = selected_ids
with open(ROOT / "results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nWROTE: stage1_predictions.csv, stage2_selection.csv/.json, "
      "stage3_simulation.csv/.json, results_summary.json")
