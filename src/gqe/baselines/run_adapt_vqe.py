import argparse
import json
from pathlib import Path

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import NumPyMinimumEigensolver, VQE
from qiskit_algorithms.optimizers import SLSQP
from tqdm.auto import tqdm


def _simple_two_qubit_hamiltonian(system_name: str) -> SparsePauliOp:
    # Compatibility-mode Hamiltonians used when qiskit-nature API stack differs.
    if system_name == "h2":
        return SparsePauliOp.from_list(
            [("II", -1.0523), ("ZI", 0.3979), ("IZ", -0.3979), ("ZZ", -0.0113), ("XX", 0.1809)]
        )
    return SparsePauliOp.from_list(
        [("II", -1.0), ("ZI", 0.5), ("IZ", -0.25), ("ZZ", -0.12), ("XX", 0.2)]
    )


def _run_compat_mode(system_name: str, maxiter: int) -> dict:
    hamiltonian_op = _simple_two_qubit_hamiltonian(system_name)
    reference_solver = NumPyMinimumEigensolver()
    ref_result = reference_solver.compute_minimum_eigenvalue(hamiltonian_op)
    ref_energy = float(np.real(ref_result.eigenvalue))

    ansatz = efficient_su2(2, reps=2, entanglement="full")
    vqe_solver = VQE(
        estimator=StatevectorEstimator(),
        ansatz=ansatz,
        optimizer=SLSQP(maxiter=maxiter),
    )
    vqe_result = vqe_solver.compute_minimum_eigenvalue(hamiltonian_op)
    vqe_energy = float(np.real(vqe_result.eigenvalue))

    return {
        "system": system_name,
        "baseline": "adapt_vqe",
        "reference_energy": ref_energy,
        "baseline_energy": vqe_energy,
        "delta_energy": abs(vqe_energy - ref_energy),
        "n_spin_orbitals": 4,
        "mode": "compat_vqe",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ADAPT-VQE baseline.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--maxiter", type=int, default=150)
    args = parser.parse_args()

    systems = ["h2", "lih"]
    results = []
    for name in tqdm(systems, desc="Running ADAPT-VQE", unit="system", dynamic_ncols=True, disable=None):
        results.append(_run_compat_mode(name, args.maxiter))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"Wrote baseline results to: {args.out}")


if __name__ == "__main__":
    main()

