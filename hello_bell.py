"""Atomiq starter kit — 1) Hello, Bell state.

Builds a 2-qubit Bell state (|00> + |11>) / sqrt(2), measures it on the
Aer simulator and prints a text histogram.

Run:  python hello_bell.py
Expected: ~50% '00', ~50% '11', ~0% '01'/'10' — the qubits are entangled,
so the two measurements always agree.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

SHOTS = 1024


def bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)            # superposition on qubit 0
    qc.cx(0, 1)        # entangle qubit 1 with qubit 0
    qc.measure([0, 1], [0, 1])
    return qc


def ascii_hist(counts: dict, shots: int) -> None:
    print("Measurement histogram (ASCII):")
    for bitstring in sorted(counts):
        n = counts[bitstring]
        bar = "#" * int(50 * n / shots)
        print(f"  {bitstring}  {n:5d}  {100 * n / shots:5.1f}%  {bar}")


def main() -> None:
    qc = bell_circuit()
    print(qc.draw(output="text"))
    sim = AerSimulator()
    counts = sim.run(qc, shots=SHOTS).result().get_counts()
    print(f"\nShots: {SHOTS}")
    print(f"Counts: {counts}")
    ascii_hist(counts, SHOTS)
    entangled = counts.get("00", 0) + counts.get("11", 0)
    print(
        f"\nEntangled outcomes (00+11): {entangled}/{SHOTS} "
        f"= {100 * entangled / SHOTS:.1f}%"
    )
    print("Expected: ~50% '00', ~50% '11', ~0% '01'/'10'.")


if __name__ == "__main__":
    main()
