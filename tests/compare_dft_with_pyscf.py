#!/usr/bin/env python3
"""Compare miniqc config-driven RKS energies against PySCF.

This script is intentionally optional.  It requires PySCF and is not enabled in
CTest unless MINIQC_ENABLE_PYSCF_TESTS=ON is passed to CMake.

The comparison tolerance is deliberately loose by default because miniqc uses a
small educational atom-centered grid and its purpose here is to catch major
implementation mistakes such as missing hybrid exchange or factor-of-two errors.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


H2_XYZ = """2
H2, R = 1.4 bohr
H  0.000000000000  0.000000000000  0.000000000000
H  0.000000000000  0.000000000000  1.400000000000
"""


PYSCF_XC_NAME = {
    "pbe": "pbe",
    "blyp": "blyp",
    "b3lyp": "b3lyp",
    "pbe0": "pbe0",
}


def parse_energy_from_miniqc_output(text: str) -> float:
    pattern = re.compile(r"^E_total\s*=\s*([-+0-9.eE]+)\s+Ha", re.MULTILINE)
    matches = pattern.findall(text)
    if not matches:
        raise RuntimeError("could not find final DFT E_total in miniqc output")
    return float(matches[-1])


def write_miniqc_input(path: Path, functional: str, xyz_name: str) -> None:
    path.write_text(
        f"""[molecule]
basis = sto-3g
xyz = {xyz_name}
unit = bohr
charge = 0
multiplicity = 1

[calculation]
job = single_point
mp2 = false
dft = true

[output]
print_orbitals = false
print_gradient = false
print_matrices = false

[scf]
verbose = false
max_iter = 128
use_diis = true

[dft]
functional = {functional}
n_radial = 40
angular_grid = 26
r_max = 10.0
radial_power = 2.0
density_mixing = 0.25
max_iter = 128
e_conv = 1.0e-8
d_conv = 1.0e-6
verbose = false
""",
        encoding="utf-8",
    )


def run_miniqc(exe: Path, workdir: Path, functional: str) -> float:
    xyz = workdir / "h2_1p4_bohr.xyz"
    xyz.write_text(H2_XYZ, encoding="utf-8")
    inp = workdir / f"h2_{functional}.in"
    write_miniqc_input(inp, functional, xyz.name)

    completed = subprocess.run(
        [str(exe), str(inp)],
        cwd=workdir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"miniqc failed for {functional} with exit code {completed.returncode}\n"
            f"{completed.stdout}"
        )
    return parse_energy_from_miniqc_output(completed.stdout)


def run_pyscf(functional: str) -> float:
    try:
        from pyscf import dft, gto
    except ImportError as exc:
        raise RuntimeError(
            "PySCF is not installed. Install it or configure CMake without "
            "-DMINIQC_ENABLE_PYSCF_TESTS=ON."
        ) from exc

    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 1.4"
    mol.unit = "Bohr"
    mol.basis = "sto-3g"
    mol.charge = 0
    mol.spin = 0
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = PYSCF_XC_NAME[functional]
    mf.grids.level = 5
    mf.conv_tol = 1.0e-10
    mf.max_cycle = 100
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"PySCF did not converge for {functional}")
    return float(energy)


def compare(functionals: Iterable[str], exe: Path, workdir: Path, tolerance: float) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for functional in functionals:
        key = functional.lower()
        if key not in PYSCF_XC_NAME:
            raise RuntimeError(f"unsupported functional for PySCF comparison: {functional}")

        miniqc_energy = run_miniqc(exe, workdir, key)
        pyscf_energy = run_pyscf(key)
        diff = miniqc_energy - pyscf_energy

        print(
            f"{key:8s}  miniqc = {miniqc_energy: .12f}  "
            f"pyscf = {pyscf_energy: .12f}  diff = {diff: .6e}"
        )

        if not math.isfinite(diff) or abs(diff) > tolerance:
            failures.append(
                f"{key}: |diff| = {abs(diff):.6e} > tolerance = {tolerance:.6e}"
            )

    if failures:
        raise RuntimeError("DFT reference comparison failed:\n" + "\n".join(failures))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miniqc-exe", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument(
        "--functionals",
        nargs="+",
        default=["pbe", "b3lyp", "pbe0"],
        help="Functionals to compare against PySCF.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5.0e-2,
        help="Absolute energy tolerance in Hartree.",
    )
    args = parser.parse_args(argv)

    compare(args.functionals, args.miniqc_exe, args.workdir, args.tolerance)
    print("PySCF DFT reference comparison passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
