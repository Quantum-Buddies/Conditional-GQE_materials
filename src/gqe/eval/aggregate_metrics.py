import argparse
import csv
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment outputs into CSV.")
    parser.add_argument("--ham", type=Path, required=True, help="Hamiltonian JSON path")
    parser.add_argument("--baseline", type=Path, required=True, help="Primary baseline JSON path")
    parser.add_argument("--cudaq-baseline", type=Path, default=None, help="Optional CUDA-Q baseline JSON path")
    parser.add_argument("--train", type=Path, required=True, help="Training metrics JSON path")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args()

    ham = _read_json(args.ham)
    baseline = _read_json(args.baseline)
    train = _read_json(args.train)

    rows = []
    for rec in ham["records"]:
        rows.append(
            {
                "section": "dataset",
                "system": rec["name"],
                "metric": "n_pauli_terms",
                "value": rec["n_pauli_terms"],
            }
        )
    def append_baseline_records(payload: dict) -> None:
        for rec in payload["results"]:
            rows.append(
                {
                    "section": "baseline",
                    "system": rec["system"],
                    "metric": f'{rec.get("baseline", "baseline")}_delta_energy',
                    "value": rec["delta_energy"],
                }
            )

    append_baseline_records(baseline)
    if args.cudaq_baseline is not None and args.cudaq_baseline.exists():
        cudaq_baseline = _read_json(args.cudaq_baseline)
        append_baseline_records(cudaq_baseline)

    rows.append(
        {
            "section": "training",
            "system": "seq_model",
            "metric": "final_loss",
            "value": train.get("final_loss"),
        }
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "system", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote aggregated CSV: {args.out}")


if __name__ == "__main__":
    main()

