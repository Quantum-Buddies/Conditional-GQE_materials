import argparse
import json
from pathlib import Path

import yaml
from openfermion import count_qubits, get_fermion_operator
from openfermion.ops import QubitOperator
from openfermion.transforms import jordan_wigner
from openfermionpyscf import generate_molecular_hamiltonian
from tqdm.auto import tqdm


def _to_serializable_terms(qubit_ham: QubitOperator) -> list[dict]:
    terms = []
    for pauli_term, coeff in qubit_ham.terms.items():
        label = " ".join([f"{p}{i}" for i, p in pauli_term]) if pauli_term else "I"
        terms.append({"term": label, "real": float(coeff.real), "imag": float(coeff.imag)})
    return terms


def generate_from_config(config_path: Path, output_dir: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for m in tqdm(
        cfg["dataset"]["molecules"],
        desc="Generating Hamiltonians",
        unit="system",
        dynamic_ncols=True,
        disable=None,
    ):
        mol_ham = generate_molecular_hamiltonian(
            geometry=m["geometry"],
            basis=cfg["dataset"]["basis"],
            multiplicity=m["multiplicity"],
            charge=m["charge"],
        )
        fermion_ham = get_fermion_operator(mol_ham)
        qubit_ham = jordan_wigner(fermion_ham)
        terms = _to_serializable_terms(qubit_ham)
        record = {
            "name": m["name"],
            "split": m["split"],
            "geometry": m["geometry"],
            "basis": cfg["dataset"]["basis"],
            "charge": m["charge"],
            "multiplicity": m["multiplicity"],
            "n_qubits": int(count_qubits(fermion_ham)),
            "n_pauli_terms": len(terms),
            "terms": terms,
        }
        records.append(record)

    out_file = output_dir / "hamiltonians.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump({"records": records}, f, indent=2)
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate molecular Hamiltonian dataset.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = generate_from_config(args.config, args.out)
    print(f"Wrote Hamiltonian dataset to: {out}")


if __name__ == "__main__":
    main()

