#!/usr/bin/env python3
"""Scan miniqc DFT grid convergence against PySCF for H2/STO-3G.

The script reuses the miniqc/PySCF helpers from compare_dft_with_pyscf.py and
prints a CSV table.  It is intended for manual validation, not normal CTest.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

from compare_dft_with_pyscf import (
    PYSCF_XC_NAME,
    normalize_functional_name,
    run_miniqc,
    run_pyscf,
)


def parse_int_list(values: list[str]) -> list[int]:
    out: list[int] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                out.append(int(token))
    return out


def parse_float_list(values: list[str]) -> list[float]:
    out: list[float] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                out.append(float(token))
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miniqc-exe", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument(
        "--functionals",
        nargs="+",
        default=["m062x"],
        help="Functionals to scan. Aliases such as m062x and M06-2X are accepted.",
    )
    parser.add_argument(
        "--n-radial",
        nargs="+",
        default=["100", "200", "300"],
        help="One or more radial grid sizes. Comma-separated values are also accepted.",
    )
    parser.add_argument(
        "--angular-grid",
        nargs="+",
        default=["302", "590", "770", "974", "1202"],
        help="One or more Lebedev angular grid sizes. Comma-separated values are also accepted.",
    )
    parser.add_argument(
        "--r-max",
        nargs="+",
        default=["10.0"],
        help="One or more radial cutoffs in bohr. Comma-separated values are also accepted.",
    )
    parser.add_argument(
        "--expected-nelec",
        type=float,
        default=2.0,
        help="Expected electron count for Ne(grid) diagnostics.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path. If omitted, CSV is written to stdout.",
    )
    args = parser.parse_args(argv)

    functionals = [normalize_functional_name(f) for f in args.functionals]
    n_radial_values = parse_int_list(args.n_radial)
    angular_values = parse_int_list(args.angular_grid)
    r_max_values = parse_float_list(args.r_max)

    for key in functionals:
        if key not in PYSCF_XC_NAME:
            raise RuntimeError(f"unsupported functional for PySCF comparison: {key}")

    args.workdir.mkdir(parents=True, exist_ok=True)

    pyscf_cache: dict[str, float] = {}
    for key in functionals:
        pyscf_cache[key] = run_pyscf(key)

    rows: list[dict[str, object]] = []
    for key in functionals:
        pyscf_energy = pyscf_cache[key]
        for r_max in r_max_values:
            for n_radial in n_radial_values:
                for angular_grid in angular_values:
                    miniqc = run_miniqc(
                        args.miniqc_exe,
                        args.workdir,
                        key,
                        n_radial,
                        angular_grid,
                        r_max,
                    )
                    diff = miniqc.energy_total - pyscf_energy
                    ne_error = miniqc.ne_grid - args.expected_nelec
                    rows.append({
                        "functional": key,
                        "n_radial": n_radial,
                        "angular_grid": angular_grid,
                        "r_max_bohr": r_max,
                        "E_miniqc_Ha": f"{miniqc.energy_total:.12f}",
                        "E_pyscf_Ha": f"{pyscf_energy:.12f}",
                        "diff_Ha": f"{diff:.9e}",
                        "abs_diff_Ha": f"{abs(diff):.9e}",
                        "Ne_grid": f"{miniqc.ne_grid:.12f}",
                        "dNe": f"{ne_error:.9e}",
                        "abs_dNe": f"{abs(ne_error):.9e}",
                    })

                    print(
                        f"{key:8s} nr={n_radial:4d} ang={angular_grid:4d} "
                        f"rmax={r_max:5.1f}  diff={diff: .6e}  "
                        f"Ne={miniqc.ne_grid:.9f}  dNe={ne_error: .3e}",
                        file=sys.stderr,
                    )

    fieldnames = [
        "functional",
        "n_radial",
        "angular_grid",
        "r_max_bohr",
        "E_miniqc_Ha",
        "E_pyscf_Ha",
        "diff_Ha",
        "abs_diff_Ha",
        "Ne_grid",
        "dNe",
        "abs_dNe",
    ]

    if args.output is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.output}", file=sys.stderr)

    if any(not math.isfinite(float(row["abs_diff_Ha"])) for row in rows):
        raise RuntimeError("non-finite energy difference encountered")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
