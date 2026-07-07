#!/usr/bin/env python3
"""Scan miniqc DFT-D dispersion energies as a function of fragment distance.

The tool generates simple noncovalent dimer geometries, runs miniqc for each
separation, and writes a CSV table containing the Kohn-Sham energy, dispersion
correction, and corrected total energy.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BOHR_PER_ANGSTROM = 1.88972612456506


@dataclass(frozen=True)
class Atom:
    symbol: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class MiniQCDFTResult:
    e_ks_total: float
    e_dispersion: float
    e_total: float
    ne_grid: float
    converged: bool
    output: str


def methane_fragment(x_shift: float) -> list[Atom]:
    """Return a tetrahedral methane fragment centered at x_shift, in Angstrom."""
    a = 0.629118  # C-H projection for r_CH = 1.089 Angstrom and tetrahedral directions.
    return [
        Atom("C", x_shift, 0.0, 0.0),
        Atom("H", x_shift + a,  a,  a),
        Atom("H", x_shift + a, -a, -a),
        Atom("H", x_shift - a,  a, -a),
        Atom("H", x_shift - a, -a,  a),
    ]


def h2_fragment(x_shift: float) -> list[Atom]:
    """Return an H2 fragment centered at x_shift, in Angstrom."""
    half = 0.370425  # 0.74085 Angstrom bond length / 2.
    return [
        Atom("H", x_shift, 0.0, -half),
        Atom("H", x_shift, 0.0,  half),
    ]


def water_fragment(x_shift: float) -> list[Atom]:
    """Return an approximately experimental water monomer centered on O, in Angstrom."""
    return [
        Atom("O", x_shift, 0.0, 0.0),
        Atom("H", x_shift + 0.757160, 0.586260, 0.0),
        Atom("H", x_shift - 0.757160, 0.586260, 0.0),
    ]


def make_dimer(model: str, separation_angstrom: float) -> list[Atom]:
    half = 0.5 * separation_angstrom
    if model == "methane":
        return methane_fragment(-half) + methane_fragment(half)
    if model == "h2":
        return h2_fragment(-half) + h2_fragment(half)
    if model == "water":
        return water_fragment(-half) + water_fragment(half)
    raise ValueError(f"unknown model: {model}")


def write_xyz(path: Path, atoms: Iterable[Atom], comment: str) -> None:
    atoms = list(atoms)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{len(atoms)}\n")
        f.write(comment + "\n")
        for a in atoms:
            f.write(f"{a.symbol:2s} {a.x: .12f} {a.y: .12f} {a.z: .12f}\n")


def write_input(
    path: Path,
    xyz_name: str,
    basis: str,
    functional: str,
    dispersion: str,
    s6: float | None,
    damping_d: float,
    n_radial: int,
    angular_grid: int,
    r_max: float,
    max_iter: int,
) -> None:
    s6_line = "" if s6 is None else f"s6 = {s6}\n"
    path.write_text(
        f"""[molecule]
basis = {basis}
xyz = {xyz_name}
unit = angstrom
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
max_iter = {max_iter}
e_conv = 1.0e-9
d_conv = 1.0e-7
verbose = false

[dispersion]
method = {dispersion}
{s6_line}damping_d = {damping_d}
""",
        encoding="utf-8",
    )


def parse_float_line(label: str, text: str) -> float:
    pattern = re.compile(rf"^{re.escape(label)}\s*=\s*([-+0-9.eE]+)(?:\s+Ha)?", re.MULTILINE)
    matches = pattern.findall(text)
    if not matches:
        raise RuntimeError(f"could not find {label!r} in miniqc output")
    return float(matches[-1])


def parse_result(text: str) -> MiniQCDFTResult:
    converged_match = re.search(r"^converged\s*=\s*(yes|no)", text, re.MULTILINE)
    converged = bool(converged_match and converged_match.group(1) == "yes")

    e_total = parse_float_line("E_total", text)
    ne_grid = parse_float_line("Ne(grid)", text)

    if re.search(r"^E_KS_total\s*=", text, re.MULTILINE):
        e_ks = parse_float_line("E_KS_total", text)
        e_disp = parse_float_line("E_dispersion", text)
    else:
        e_ks = e_total
        e_disp = 0.0

    return MiniQCDFTResult(
        e_ks_total=e_ks,
        e_dispersion=e_disp,
        e_total=e_total,
        ne_grid=ne_grid,
        converged=converged,
        output=text,
    )


def run_miniqc(exe: Path, inp_name: str, workdir: Path) -> MiniQCDFTResult:
    completed = subprocess.run(
        [str(exe), inp_name],
        cwd=workdir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"miniqc failed for {inp_name} with exit code {completed.returncode}\n"
            f"Command: {exe} {inp_name}\n"
            f"Working directory: {workdir}\n"
            f"{completed.stdout}"
        )
    return parse_result(completed.stdout)


def parse_distances(values: list[str]) -> list[float]:
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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model",
        choices=["methane", "water", "h2"],
        default="methane",
        help="Built-in dimer model to scan.",
    )
    parser.add_argument(
        "--distances",
        nargs="+",
        default=["3.5", "4.0", "4.5", "5.0", "6.0", "7.0", "8.0", "10.0"],
        help="Center-to-center distances in Angstrom. Comma-separated values are accepted.",
    )
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--functional", default="pbe")
    parser.add_argument("--dispersion", default="d2", choices=["none", "d2"])
    parser.add_argument("--s6", type=float, default=None, help="Optional manual D2 s6 scale factor.")
    parser.add_argument("--damping-d", type=float, default=20.0)
    parser.add_argument("--n-radial", type=int, default=80)
    parser.add_argument("--angular-grid", type=int, default=50)
    parser.add_argument("--r-max", type=float, default=12.0)
    parser.add_argument("--max-iter", type=int, default=128)
    args = parser.parse_args(argv)

    distances = parse_distances(args.distances)
    exe = args.miniqc_exe.expanduser().resolve()
    workdir = args.workdir.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not exe.exists():
        raise FileNotFoundError(f"miniqc executable not found: {exe}")

    workdir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for distance in distances:
        tag = f"{args.model}_{distance:.3f}".replace(".", "p")
        xyz = workdir / f"{tag}.xyz"
        inp = workdir / f"{tag}.in"

        atoms = make_dimer(args.model, distance)
        write_xyz(xyz, atoms, f"{args.model} dimer, center separation = {distance:.6f} Angstrom")
        write_input(
            inp,
            xyz.name,
            args.basis,
            args.functional,
            args.dispersion,
            args.s6,
            args.damping_d,
            args.n_radial,
            args.angular_grid,
            args.r_max,
            args.max_iter,
        )

        result = run_miniqc(exe, inp.name, workdir)
        rows.append({
            "model": args.model,
            "distance_angstrom": f"{distance:.6f}",
            "distance_bohr": f"{distance * BOHR_PER_ANGSTROM:.9f}",
            "basis": args.basis,
            "functional": args.functional,
            "dispersion": args.dispersion,
            "E_KS_total_Ha": f"{result.e_ks_total:.12f}",
            "E_dispersion_Ha": f"{result.e_dispersion:.12f}",
            "E_total_Ha": f"{result.e_total:.12f}",
            "E_dispersion_mHa": f"{1000.0 * result.e_dispersion:.9f}",
            "Ne_grid": f"{result.ne_grid:.12f}",
            "converged": "yes" if result.converged else "no",
            "xyz": xyz.name,
            "input": inp.name,
        })

        print(
            f"R={distance:6.3f} A  "
            f"E_KS={result.e_ks_total: .9f}  "
            f"E_disp={result.e_dispersion: .9e}  "
            f"E_total={result.e_total: .9f}",
            file=sys.stderr,
        )

    fieldnames = [
        "model",
        "distance_angstrom",
        "distance_bohr",
        "basis",
        "functional",
        "dispersion",
        "E_KS_total_Ha",
        "E_dispersion_Ha",
        "E_total_Ha",
        "E_dispersion_mHa",
        "Ne_grid",
        "converged",
        "xyz",
        "input",
    ]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
