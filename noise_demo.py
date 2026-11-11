"""Atomiq starter kit — 3) Noise as a feature: Bell state, ideal vs noisy.

Runs the same Bell-state circuit twice on Aer:
  * ideal simulator (no noise) — ~50% '00', ~50% '11'
  * noisy simulator — depolarising error on 1-qubit gates (1%) and
    2-qubit gates (5%) plus 3% readout error

The point is NOT that noise is good. The talking point is:
  * noise is measurable — the '01'/'10' outcomes that should never appear
    are exactly the error signal;
  * noise is localisable — which gate/qubit the errors come from tells you
    where to spend error-mitigation effort (readout calibration, dynamical
    decoupling, zero-noise extrapolation);
  * on real hardware the same experiment is a calibration diagnostic, not
    just a demo.

Run:  python noise_demo.py
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

SHOTS = 1024


def bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def noisy_model() -> NoiseModel:
    model = NoiseModel()
    # 1% depolarising error on every 1-qubit gate, 5% on 2-qubit gates.
    model.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ["h"])
    model.add_all_qubit_quantum_error(depolarizing_error(0.05, 2), ["cx"])
    # 3% chance a 0 reads as 1 and vice versa.
    model.add_all_qubit_readout_error(ReadoutError([[0.97, 0.03], [0.03, 0.97]]))
    return model


def entangled_fraction(counts: dict) -> float:
    return (counts.get("00", 0) + counts.get("11", 0)) / sum(counts.values())


def main() -> None:
    qc = bell_circuit()

    ideal = AerSimulator().run(qc, shots=SHOTS).result().get_counts()
    noisy = (
        AerSimulator(noise_model=noisy_model())
        .run(qc, shots=SHOTS)
        .result()
        .get_counts()
    )

    print(f"Shots per run: {SHOTS}\n")
    print(f"IDEAL : {dict(sorted(ideal.items()))}")
    print(
        f"        entangled outcomes (00+11): "
        f"{100 * entangled_fraction(ideal):.1f}%"
    )
    print(f"NOISY : {dict(sorted(noisy.items()))}")
    print(
        f"        entangled outcomes (00+11): "
        f"{100 * entangled_fraction(noisy):.1f}%"
    )
    bad = noisy.get("01", 0) + noisy.get("10", 0)
    print(
        f"        impossible outcomes (01+10): {bad} shots "
        f"({100 * bad / SHOTS:.1f}%) — this IS the error signal"
    )
    print(
        "\nTakeaway: the same circuit, two realities. The gap between "
        "them is the demo — measure it, name the dominant error source "
        "(here: the 5% two-qubit gate error + readout), and show the "
        "mitigation you would apply on real hardware."
    )


if __name__ == "__main__":
    main()
