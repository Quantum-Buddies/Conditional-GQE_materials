import argparse
import json
from pathlib import Path

import cudaq
import cudaq_solvers as solvers
import numpy as np
from scipy.optimize import minimize


def _run_h2(maxiter: int, method: str = "COBYLA") -> dict:
    geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7474))]
    molecule = solvers.create_molecule(geometry=geometry, basis="sto-3g", spin=0, charge=0, casci=True)

    num_qubits = int(molecule.n_orbitals * 2)
    num_electrons = int(molecule.n_electrons)
    spin = 0
    n_params = int(solvers.stateprep.get_num_uccsd_parameters(num_electrons, num_qubits, spin))
    initial_x = [-0.2] * n_params

    @cudaq.kernel
    def ansatz(thetas: list[float]):
        q = cudaq.qvector(num_qubits)
        for i in range(num_electrons):
            x(q[i])
        solvers.stateprep.uccsd(q, thetas, num_electrons, spin)

    # Match NVIDIA documentation VQE pattern
    # Multiple optimizers are supported (e.g. COBYLA, L-BFGS-B)
    vqe_kwargs = {
        "optimizer": minimize,
        "method": method,
        "options": {"maxiter": maxiter},
    }
    
    if method == "L-BFGS-B":
        vqe_kwargs["jac"] = "3-point"
        vqe_kwargs["tol"] = 1e-4

    energy, _, _ = solvers.vqe(
        ansatz,
        molecule.hamiltonian,
        initial_x,
        **vqe_kwargs
    )

    # FCI/CASCI reference exposed by create_molecule result when casci=True.
    reference_energy = float(molecule.energies["fci_energy"] if "fci_energy" in molecule.energies else energy)
    baseline_energy = float(np.real(energy))
    return {
        "system": "h2",
        "baseline": "cudaq_vqe",
        "reference_energy": reference_energy,
        "baseline_energy": baseline_energy,
        "delta_energy": abs(baseline_energy - reference_energy),
        "n_spin_orbitals": num_qubits,
        "mode": "cudaq_uccsd_vqe",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CUDA-Q VQE baseline.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--method", type=str, default="COBYLA", help="Optimizer method (e.g., COBYLA, L-BFGS-B)")
    args = parser.parse_args()

    result = _run_h2(maxiter=args.maxiter, method=args.method)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"results": [result]}, f, indent=2)
    print(f"Wrote CUDA-Q baseline results to: {args.out}")


if __name__ == "__main__":
    main()

