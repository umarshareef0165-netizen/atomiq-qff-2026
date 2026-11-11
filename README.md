# Atomiq — Hybrid Quantum-Classical Term Deposit Campaign Pipeline

**The Atomiq Hackathon | Qiskit Fall Fest 2026 at IEM** — theme: *Chi(r)ps: A Decade of Atomiq Harmony on the Cloud*
**Team:** solo — Umar Shareef (UAF BSCS). **Deadline:** Wed Nov 11, 2026, 8:30 AM PT (10:00 PM IST).
**Status:** pipeline complete and verified. The only steps left are the submitter's own:
GitHub push, deck link, 1–2 min video, Devpost submit — see `SUBMISSION_CHECKLIST.md`.

## Final answer

- **Contact these 6 customers:** C0031, C0079, C0256, C0338, C0170, C0327
- **Expected net value:** 33.2167 (QAOA = brute-force optimum, **optimality gap 0.0**)
- **Totals:** contact cost 10 / 25 budget · capacity 15 / 22 · exactly 6 selected
- **P(at least 3 of the 6 subscribe):** quantum **0.3574** (16,384 shots, Aer) ·
  exact Poisson-binomial **0.3603** · Monte Carlo (200k) **0.3595** · noisy Aer **0.3787**

## The problem (official brief)

A bank runs a term-deposit campaign in three linked stages:

1. **Predict** subscription probability for 100 test customers from 300 labelled ones.
2. **Select** exactly 6 of the top 10 by predicted probability, maximising expected
   net value under a budget (total contact cost ≤ 25) and a capacity limit
   (total capacity units ≤ 22). Expected net value of customer *i*:
   `p_i × expected_deposit_value_i − contact_cost_i`.
3. **Simulate** the probability that at least 3 of the 6 selected customers subscribe
   (at least 50% success).

Data: `resources/bank_campaign_participant_release.xlsx` (official hackathon Drive
folder) — sampled from `bank.csv`, 400 customers, `duration` removed, seeds 42.
Stage 1 train 300 (35 yes / 265 no), test 100 (labels withheld). Stage 2 data
(contact cost, expected deposit value, capacity units, segment) for all 400.
Full statement: `resources/PROBLEM_STATEMENT_THE_ATOMIQ_HACKTHON.pdf`.

## Stage 1 — prediction (hybrid classifier)

**Preprocessing (fit on train only, no leakage):** standard-scaled numerics
(age, balance, day, campaign, pdays, previous) + one-hot categoricals
(job, marital, education, default, housing, loan, contact, month, poutcome).

**Models, validated by 5-fold stratified CV on train only** (out-of-fold predictions;
test never used for selection or tuning):

| Model | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---|---|---|---|
| Logistic regression | 0.6286 | 0.2001 | 0.2025 | 0.5955 |
| **Random forest (submitted)** | **0.6467** | **0.2331** | **0.1011** | **0.3555** |
| Gradient boosting | 0.5663 | 0.1924 | 0.1140 | 0.5240 |
| Quantum kernel SVC | 0.4905 | 0.1204 | 0.1034 | 0.3622 |
| Hybrid (logreg + Q-kernel) | 0.6293 | 0.1972 | 0.1228 | 0.4093 |
| Hybrid (RF + Q-kernel) | 0.6437 | 0.2221 | 0.1003 | 0.3474 |

**Quantum classifier:** 4-qubit ZZ feature map (reps 2, linear entanglement) on
pdays, campaign, balance, day — chosen by train-only mutual information — with a
fidelity kernel `K_ij = |⟨φ(x_i)|φ(x_j)⟩|²` computed exactly from statevectors
(cross-checked against `qiskit-machine-learning`'s `FidelityQuantumKernel` on a
6×6 subset: max difference < 1e-6), then a probability-calibrated SVC.
**Honest result:** on these four features the quantum kernel carries almost no
signal (ROC-AUC 0.49, PR-AUC 0.12 vs 0.117 base rate). It is built, validated and
reported — not hidden — but it is not the submitted model.

**Final-model rule (pre-registered, train OOF only):** best PR-AUC among all six
candidates, Brier as tie-break → **random forest**. The submitted probabilities
are the random forest's, clipped to [0.001, 0.999].

**Top 10** (by predicted probability, ties broken by customer_id ascending):

| Rank | Customer | p | Rank | Customer | p |
|---|---|---|---|---|---|
| 1 | C0031 | 0.4200 | 6 | C0183 | 0.3300 |
| 2 | C0079 | 0.4133 | 7 | C0170 | 0.3200 |
| 3 | C0081 | 0.3733 | 8 | C0378 | 0.2833 |
| 4 | C0256 | 0.3500 | 9 | C0327 | 0.2767 |
| 5 | C0338 | 0.3433 | 10 | C0188 | 0.2600 |

Full list: `stage1_predictions.csv` (100 rows).

## Stage 2 — selection (QAOA, checked against brute force)

**Optimisation:** maximise `Σ (p_i · deposit_i − cost_i) · x_i` over the top 10,
subject to `Σx_i = 6`, `Σcost_i·x_i ≤ 25`, `Σcap_i·x_i ≤ 22`.

**A verified simplification:** over all 210 six-subsets, the largest possible
total contact cost is **16** (< 25) and the largest possible total capacity is
**18** (< 22) — so the two inequalities are automatically satisfied by every
6-subset and never bind. The QUBO therefore enforces only the cardinality
constraint, with penalty `P·(Σx − 6)²`, `P = 15` (a one-unit violation costs 15,
more than the largest single net value, 6.92; the thinnest margin is a
7-subset at 15 + 33.22 − 40.14 = 8.08). Cost/capacity feasibility is still
checked on every sampled bitstring. Building the QUBO by hand avoids the
converter's slack variables, which would inflate the problem to 20 qubits for
constraints that never bind.

**QAOA:** 10-qubit ansatz built from plain RZZ/RZ/RX gates (cost layer, then
X-mixer, per rep) — the same unitary as `QAOAAnsatz`, verified statevector-equal
to 1e-16 on a test operator, but ~100× faster to evaluate because no
`PauliEvolutionGate` decomposition runs on every optimiser step. Optimised with
scipy COBYLA (4 fixed initialisations, 400 iterations each), exact statevector
expectation against the precomputed diagonal QUBO energies. Final circuits
sampled on Aer (4,096 shots, seed 42).

**Results:**

| | p = 2 | p = 3 |
|---|---|---|
| Optimised expectation (QUBO energy) | −13.23 | −14.79 |
| Most-probable state | the optimum (Z = 33.2167) | suboptimal (Z = 20.93) |
| P(brute-force optimum) in the state | 0.0026 | 0.0000 |

**Honest reading:** at p = 2 the most-probable state *is* the brute-force optimum,
so the submitted selection has **optimality gap 0.0** — but its probability is
only 0.26% and the sampled distribution is very flat (top sampled strings ≈
19/4,096 shots each), and at p = 3 the optimiser concentrates (12% of shots) on
a suboptimal state instead. QAOA found the optimum here; it did not robustly
converge to it. The optimum itself is certain: exact brute force over all 210
combinations gives the same six customers, Z = 33.2167, cost 10, capacity 15.
Full detail: `stage2_selection.csv` / `stage2_selection.json` (top-10 candidates,
net values, sampled bitstrings, both QAOA runs).

## Stage 3 — outcome simulation (6-qubit circuit)

One qubit per selected customer, in this order: qubit 0 = C0031, 1 = C0079,
2 = C0256, 3 = C0338, 4 = C0170, 5 = C0327. Each qubit gets
`RY(2·arcsin(√p_i))`, so `P(|1⟩) = p_i`; customers are treated as independent.
Success = at least 3 ones measured.

| Customer | p | RY angle (rad) |
|---|---|---|
| C0031 | 0.4200 | 1.4101 |
| C0079 | 0.4133 | 1.3966 |
| C0256 | 0.3500 | 1.2661 |
| C0338 | 0.3433 | 1.2521 |
| C0170 | 0.3200 | 1.2025 |
| C0327 | 0.2767 | 1.1078 |

| Estimate | P(≥3 subscribe) |
|---|---|
| Quantum (Aer, 16,384 shots) | **0.3574** |
| Exact Poisson-binomial (dynamic programming) | 0.3603 |
| Monte Carlo (200,000 trials, seed 42) | 0.3595 |
| Noisy Aer (1% 1-qubit + 5% 2-qubit depolarising, 3% readout) | 0.3787 |

Errors: quantum − exact = **−0.0028** (within the ±0.0038 shot-noise scale at
16,384 shots); Monte Carlo − exact = −0.0008; noisy − exact = +0.0185.
Transpiled circuit: depth 2, 6 RY + 6 measurements, 0 SWAPs, 6 logical /
6 physical qubits (a product state — no entangling gates are needed here, and
none are claimed). Full detail: `stage3_simulation.csv` / `stage3_simulation.json`.

## Error analysis and limitations

- **Stage 1 is the bottleneck.** Only 35 positive examples in 300 training rows,
  so ranking is modest (best PR-AUC 0.23 vs 0.117 base rate) — and Stages 2–3
  inherit those probabilities unchanged. Test labels are withheld, so the
  reported metrics are train cross-validation, not test scores.
- **Stage 2:** QAOA is heuristic; the brute-force check is what makes the
  optimum certain. The penalty landscape is rugged and the optimiser's
  concentration is weak (see p = 2 vs p = 3 above).
- **Stage 3:** finite-shot error (~±0.004) plus readout/depolarising noise in
  the noisy run; both reported next to the exact result. Independence of
  customers is an assumption of the brief's model, not a measured fact.
- **No quantum-advantage claim.** Everything runs on the Aer simulator on a
  small problem; classical references match or beat the quantum steps. The
  quantum parts are demonstrated, validated against exact classical references,
  and kept separate from the classical baselines.

## Business impact

Contact C0031, C0079, C0256, C0338, C0170, C0327 — total cost 10 of a 25 budget,
capacity 15 of 22, expected net value 33.22. The chance at least 3 subscribe is
~36%: the campaign target is a coin-flip, not a certainty — do not over-commit
follow-up capacity. The other top-10 names cost more per expected deposit than
the six chosen once the selection is optimised. Next step: re-score weekly as
new outcomes arrive, refit on train only, and re-run this same pipeline.

## Reproducibility

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_pipeline.py   # regenerates every result file (~1 min)
.venv/bin/python make_deck.py     # regenerates deck.pdf (8 slides)
```

- **Seeds:** 42 for the data split, model CV, QAOA sampling, Aer shots,
  transpiler, and Monte Carlo. Nothing is random beyond the seed.
- **Backends:** AerSimulator only (no IBM Quantum account, no hardware run).
- **Versions (pinned):** Qiskit 2.6.2, qiskit-aer 0.18.2,
  qiskit-machine-learning 0.9.1, qiskit-optimization 0.7.0,
  qiskit-algorithms 0.4.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.4,
  scipy 1.18.1, openpyxl 3.1.5, matplotlib 3.11.2.
- **Files:** `run_pipeline.py` (all three stages), `make_deck.py` (slides),
  `resources/` (official dataset + problem statement), `stage1_predictions.csv`,
  `stage2_selection.csv` + `.json`, `stage3_simulation.csv` + `.json`,
  `results_summary.json`, `deck.pdf`, `video_script.md`, `SUBMISSION_CHECKLIST.md`.
  `_kernel_cache.npz` is a cached fidelity-kernel matrix (regenerable).
- **Licence:** MIT (`LICENSE`).
