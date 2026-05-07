"""CUDA-Q Generative Quantum Eigensolver (GQE) baseline — H2 smoke example."""

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# cudaq-solvers[gqe] imports PyTorch; on HPC nodes the bundled PyTorch CUDA
# build may not match the host driver, which emits a noisy UserWarning even
# when CUDA-Q itself runs fine. Silence only these known messages.
warnings.filterwarnings(
    "ignore",
    message=r".*NVIDIA driver on your system is too old.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*CUDA initialization:.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*PyTorch wheel with CUDA.*",
    category=UserWarning,
)

for _log in ("torch", "torch.cuda"):
    logging.getLogger(_log).setLevel(logging.ERROR)

try:
    import cudaq  # type: ignore[import-untyped]
    import cudaq_solvers as solvers  # type: ignore[import-untyped]
    from cudaq import spin  # type: ignore[import-untyped]
    from cudaq_solvers.gqe_algorithm.gqe import (  # type: ignore[import-untyped]
        get_default_config,
    )
except Exception as exc:  # pragma: no cover - handled at runtime
    cudaq = None  # type: ignore[assignment]
    solvers = None  # type: ignore[assignment]
    spin = None  # type: ignore[assignment]
    get_default_config = None  # type: ignore[assignment]
    _CUDAQ_IMPORT_ERROR: Optional[Exception] = exc
else:
    _CUDAQ_IMPORT_ERROR = None


def _ensure_cudaq_available() -> None:
    if cudaq is None or solvers is None or get_default_config is None:
        msg = (
            "CUDA-Q GQE baseline requires `cudaq` and `cudaq-solvers[gqe]`.\n"
            "Install into the experiment environment, for example:\n"
            "  pip install cudaq-solvers[gqe]\n"
        )
        raise RuntimeError(msg) from _CUDAQ_IMPORT_ERROR


def _serialize_selected_operators(
    op_pool: List[Any], indices: List[int], n_qubits: int
) -> List[Dict[str, Any]]:
    """Serialize selected operators into a JSON-friendly structure."""
    records: List[Dict[str, Any]] = []
    for idx in indices:
        op = op_pool[idx]
        terms: List[Dict[str, Any]] = []
        for term in op:
            coeff = term.evaluate_coefficient()
            try:
                pw = term.get_pauli_word(n_qubits)
                pw_str = str(pw)
            except Exception:  # pragma: no cover - defensive
                pw_str = None
            terms.append(
                {
                    "coefficient_real": float(coeff.real),
                    "coefficient_imag": float(coeff.imag),
                    "pauli_word": pw_str,
                }
            )
        records.append({"index": int(idx), "terms": terms})
    return records


def _run_h2_gqe(max_iters: int, ngates: int) -> Dict[str, Any]:
    """Run CUDA-Q GQE on the H2 molecule following NVIDIA's example pattern."""
    _ensure_cudaq_available()
    if not hasattr(cudaq, "pauli_word"):  # type: ignore[union-attr]
        raise RuntimeError(
            "Installed CUDA-Q is missing `cudaq.pauli_word`, which is required "
            "for GQE kernel argument typing in this script. Please upgrade CUDA-Q."
        )

    # Choose a target consistent with the official examples, but fall back to CPU
    # if NVIDIA hardware or the fp64 option is unavailable.
    try:
        cudaq.set_target("nvidia", option="fp64")  # type: ignore[union-attr]
    except RuntimeError:
        cudaq.set_target("qpp-cpu")  # type: ignore[union-attr]

    # Molecular Hamiltonian (same geometry as the CUDA-Q VQE baseline)
    geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7474))]
    molecule = solvers.create_molecule(  # type: ignore[union-attr]
        geometry=geometry,
        basis="sto-3g",
        spin=0,
        charge=0,
        casci=True,
    )

    spin_ham = molecule.hamiltonian
    n_qubits = int(molecule.n_orbitals * 2)
    n_electrons = int(molecule.n_electrons)

    # Operator pool and cost follow the NVIDIA `gqe_h2.py` example, but we keep
    # the configuration minimal (no CSV logging, no MPI path).
    params = [
        0.003125,
        -0.003125,
        0.00625,
        -0.00625,
        0.0125,
        -0.0125,
        0.025,
        -0.025,
        0.05,
        -0.05,
        0.1,
        -0.1,
    ]

    def build_pool() -> List[Any]:
        ops: List[Any] = []
        i = 0

        ops.append(
            cudaq.SpinOperator(  # type: ignore[union-attr]
                spin.y(i) * spin.z(i + 1) * spin.x(i + 2) * spin.i(i + 3)
            )
        )
        ops.append(
            cudaq.SpinOperator(
                spin.x(i) * spin.z(i + 1) * spin.y(i + 2) * spin.i(i + 3)
            )
        )
        ops.append(
            cudaq.SpinOperator(
                spin.i(i) * spin.y(i + 1) * spin.z(i + 2) * spin.x(i + 3)
            )
        )
        ops.append(
            cudaq.SpinOperator(
                spin.i(i) * spin.x(i + 1) * spin.z(i + 2) * spin.y(i + 3)
            )
        )
        ops.append(
            cudaq.SpinOperator(
                spin.x(i) * spin.x(i + 1) * spin.x(i + 2) * spin.y(i + 3)
            )
        )
        ops.append(
            cudaq.SpinOperator(
                spin.x(i) * spin.x(i + 1) * spin.y(i + 2) * spin.x(i + 3)
            )
        )
        ops.append(
            cudaq.SpinOperator(
                spin.x(i) * spin.y(i + 1) * spin.y(i + 2) * spin.y(i + 3)
            )
        )
        ops.append(
            cudaq.SpinOperator(
                spin.y(i) * spin.x(i + 1) * spin.y(i + 2) * spin.y(i + 3)
            )
        )
        ops.append(
            cudaq.SpinOperator(
                spin.x(i) * spin.y(i + 1) * spin.x(i + 2) * spin.x(i + 3)
            )
        )
        ops.append(
            cudaq.SpinOperator(
                spin.y(i) * spin.x(i + 1) * spin.x(i + 2) * spin.x(i + 3)
            )
        )
        ops.append(
            cudaq.SpinOperator(
                spin.y(i) * spin.y(i + 1) * spin.x(i + 2) * spin.y(i + 3)
            )
        )
        ops.append(
            cudaq.SpinOperator(
                spin.y(i) * spin.y(i + 1) * spin.y(i + 2) * spin.x(i + 3)
            )
        )

        pool: List[Any] = []
        for c in params:
            for op in ops:
                pool.append(c * op)
        return pool

    op_pool = build_pool()

    def term_coefficients(op: Any) -> List[complex]:
        return [term.evaluate_coefficient() for term in op]

    def term_words(op: Any) -> List[cudaq.pauli_word]:
        return [term.get_pauli_word(n_qubits) for term in op]

    @cudaq.kernel  # type: ignore[union-attr]
    def kernel(
        n_qubits_kernel: int,
        n_electrons_kernel: int,
        coeffs: List[float],
        words: List[cudaq.pauli_word],
    ) -> None:
        q = cudaq.qvector(n_qubits_kernel)  # type: ignore[union-attr]

        for i in range(n_electrons_kernel):
            x(q[i])  # type: ignore[name-defined]

        for j in range(len(coeffs)):
            exp_pauli(coeffs[j], q, words[j])  # type: ignore[name-defined]

    def cost(sampled_ops: List[Any], **_: Any) -> float:
        full_coeffs: List[float] = []
        full_words: List[cudaq.pauli_word] = []

        for op in sampled_ops:
            full_coeffs += [c.real for c in term_coefficients(op)]
            full_words += term_words(op)

        result = cudaq.observe(  # type: ignore[union-attr]
            kernel,
            spin_ham,
            n_qubits,
            n_electrons,
            full_coeffs,
            full_words,
        )
        return float(result.expectation())

    cfg = get_default_config()  # type: ignore[operator]
    cfg.use_fabric_logging = False
    cfg.save_trajectory = False
    cfg.verbose = False

    min_energy, best_indices = solvers.gqe(  # type: ignore[union-attr]
        cost,
        op_pool,
        max_iters=max_iters,
        ngates=ngates,
        config=cfg,
    )

    reference_energy = float(
        molecule.energies.get("fci_energy", min_energy)  # type: ignore[index]
    )
    baseline_energy = float(min_energy)

    return {
        "system": "h2",
        "baseline": "cudaq_gqe",
        "reference_energy": reference_energy,
        "baseline_energy": baseline_energy,
        "delta_energy": abs(baseline_energy - reference_energy),
        "n_spin_orbitals": n_qubits,
        "mode": "cudaq_gqe",
        "gqe_config": {
            "max_iters": max_iters,
            "ngates": ngates,
            "num_samples": int(getattr(cfg, "num_samples", 5)),
        },
        "gqe_selected_operators": _serialize_selected_operators(
            op_pool, list(map(int, best_indices)), n_qubits
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CUDA-Q GQE baseline.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--max-iters",
        type=int,
        default=25,
        help="Number of GQE epochs (cfg.max_iters).",
    )
    parser.add_argument(
        "--ngates",
        type=int,
        default=10,
        help="Number of gates per generated circuit (cfg.ngates).",
    )
    args = parser.parse_args()

    result = _run_h2_gqe(max_iters=args.max_iters, ngates=args.ngates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"results": [result]}, f, indent=2)
    print(f"Wrote CUDA-Q GQE baseline results to: {args.out}")


if __name__ == "__main__":
    main()

