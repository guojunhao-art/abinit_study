#!/usr/bin/env python3
"""Compare miniqc config-driven RKS energies against PySCF.

This script is intentionally optional.  It requires PySCF and is not enabled in
CTest unless MINIQC_ENABLE_PYSCF_TESTS=ON is passed to CMake.

The comparison is designed as a numerical implementation check for the DFT path:

    config -> XCFunctional -> XCEvaluator -> dft_xc_matrix -> run_rks_xc

For H2/STO-3G, a sufficiently dense miniqc grid reproduces PySCF PBE/B3LYP/PBE0
energies to around 1e-10 Ha in the tested environment.  The default tolerance is
kept at 1e-6 Ha to avoid overfitting to a specific Libxc/PySCF/BLAS build.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from dataclasses import dataclass
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
    "m062x": "M06-2X",
}


@dataclass(frozen=True)
class MiniQCResult:
    energy_total: float
    ne_grid: float
    output: str


@dataclass(frozen=True)
class ComparisonResult:
    functional: str
    miniqc_energy: float
    pyscf_energy: float
    energy_diff: float
    ne_grid: float
    ne_error: float


def normalize_functional_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def parse_miniqc_output(text: str) -> MiniQCResult:
    energy_pattern = re.compile(r"^E_total\s*=\s*([-+0-9.eE]+)\s+Ha", re.MULTILINE)
    ne_pattern = re.compile(r"^Ne\(grid\)\s*=\s*([-+0-9.eE]+)", re.MULTILINE)

    energy_matches = energy_pattern.findall(text)
    if not energy_matches:
        raise RuntimeError("could not find final DFT E_total in miniqc output")

    ne_matches = ne_pattern.findall(text)
    if not ne_matches:
        raise RuntimeError("could not find Ne(grid) in miniqc output")

    return MiniQCResult(
        energy_total=float(energy_matches[-1]),
        ne_grid=float(ne_matches[-1]),
        output=text,
    )


def parse_energy_from_miniqc_output(text: str) -> float:
    """Backward-compatible helper for older callers."""
    return parse_miniqc_output(text).energy_total


def write_miniqc_input(
    path: Path,
    functional: str,
    xyz_name: str,
    n_radial: int,
    angular_grid: int,
    r_max: float,
) -> None:
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
n_radial = {n_radial}
angular_grid = {angular_grid}
r_max = {r_max}
radial_power = 2.0
density_mixing = 0.25
max_iter = 128
e_conv = 1.0e-10
d_conv = 1.0e-8
verbose = false
""",
        encoding="utf-8",
    )


def run_miniqc(
    exe: Path,
    workdir: Path,
    functional: str,
    n_radial: int,
    angular_grid: int,
    r_max: float,
) -> MiniQCResult:
    xyz = workdir / "h2_1p4_bohr.xyz"
    xyz.write_text(H2_XYZ, encoding="utf-8")

    key = normalize_functional_name(functional)
    inp = workdir / f"h2_{key}_nr{n_radial}_ang{angular_grid}_r{r_max:g}.in"
    write_miniqc_input(inp, functional, xyz.name, n_radial, angular_grid, r_max)

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
    return parse_miniqc_output(completed.stdout)


def run_pyscf(functional: str) -> float:
    try:
        from pyscf import dft, gto
    except ImportError as exc:
        raise RuntimeError(
            "PySCF is not installed. Install it or configure CMake without "
            "-DMINIQC_ENABLE_PYSCF_TESTS=ON."
        ) from exc

    key = normalize_functional_name(functional)
    if key not in PYSCF_XC_NAME:
        raise RuntimeError(f"unsupported functional for PySCF comparison: {functional}")

    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 1.4"
    mol.unit = "Bohr"
    mol.basis = "sto-3g"
    mol.charge = 0
    mol.spin = 0
    # miniqc currently sets Libint basis shells to Cartesian form.  Keep the
    # PySCF reference in the same convention so later p/d-shell tests compare
    # the same AO basis representation.
    mol.cart = True
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = PYSCF_XC_NAME[key]
    mf.grids.level = 5
    mf.conv_tol = 1.0e-12
    mf.max_cycle = 100
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"PySCF did not converge for {functional}")
    return float(energy)


def compare(
    functionals: Iterable[str],
    exe: Path,
    workdir: Path,
    tolerance: float,
    n_radial: int,
    angular_grid: int,
    r_max: float,
    expected_nelec: float = 2.0,
    ne_tolerance: float | None = None,
) -> list[ComparisonResult]:
    workdir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    results: list[ComparisonResult] = []

    print(
        "miniqc grid: "
        f"n_radial = {n_radial}, angular_grid = {angular_grid}, r_max = {r_max} bohr"
    )
    print(f"absolute energy tolerance = {tolerance:.3e} Ha")
    if ne_tolerance is not None:
        print(f"absolute Ne(grid) tolerance = {ne_tolerance:.3e}")

    for functional in functionals:
        key = normalize_functional_name(functional)
        if key not in PYSCF_XC_NAME:
            raise RuntimeError(f"unsupported functional for PySCF comparison: {functional}")

        miniqc = run_miniqc(exe, workdir, key, n_radial, angular_grid, r_max)
        pyscf_energy = run_pyscf(key)
        diff = miniqc.energy_total - pyscf_energy
        ne_error = miniqc.ne_grid - expected_nelec

        result = ComparisonResult(
            functional=key,
            miniqc_energy=miniqc.energy_total,
            pyscf_energy=pyscf_energy,
            energy_diff=diff,
            ne_grid=miniqc.ne_grid,
            ne_error=ne_error,
        )
        results.append(result)

        print(
            f"{key:8s}  miniqc = {miniqc.energy_total: .12f}  "
            f"pyscf = {pyscf_energy: .12f}  diff = {diff: .6e}  "
            f"Ne = {miniqc.ne_grid: .9f}  dNe = {ne_error: .3e}"
        )

        if not math.isfinite(diff) or abs(diff) > tolerance:
            failures.append(
                f"{key}: |diff| = {abs(diff):.6e} > tolerance = {tolerance:.6e}"
            )
        if ne_tolerance is not None:
            if not math.isfinite(ne_error) or abs(ne_error) > ne_tolerance:
                failures.append(
                    f"{key}: |dNe| = {abs(ne_error):.6e} > tolerance = {ne_tolerance:.6e}"
                )

    if failures:
        raise RuntimeError("DFT reference comparison failed:\n" + "\n".join(failures))

    return results


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miniqc-exe", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument(
        "--functionals",
        nargs="+",
        default=["pbe", "b3lyp", "pbe0"],
        help="Functionals to compare against PySCF. Aliases such as m062x and M06-2X are accepted.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0e-6,
        help="Absolute energy tolerance in Hartree.",
    )
    parser.add_argument(
        "--ne-tolerance",
        type=float,
        default=None,
        help="Optional absolute tolerance for Ne(grid). If omitted, Ne(grid) is reported but not enforced.",
    )
    parser.add_argument(
        "--expected-nelec",
        type=float,
        default=2.0,
        help="Expected electron count for Ne(grid) diagnostics.",
    )
    parser.add_argument(
        "--n-radial",
        type=int,
        default=200,
        help="miniqc atom-centered radial grid size.",
    )
    parser.add_argument(
        "--angular-grid",
        type=int,
        default=302,
        help="miniqc Lebedev angular grid size.",
    )
    parser.add_argument(
        "--r-max",
        type=float,
        default=10.0,
        help="miniqc radial grid cutoff in bohr.",
    )
    args = parser.parse_args(argv)

    compare(
        args.functionals,
        args.miniqc_exe,
        args.workdir,
        args.tolerance,
        args.n_radial,
        args.angular_grid,
        args.r_max,
        expected_nelec=args.expected_nelec,
        ne_tolerance=args.ne_tolerance,
    )
    print("PySCF DFT reference comparison passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
