# abinit_study

A small C++ learning project for implementing basic **ab initio electronic-structure methods** from scratch.

This repository is intended for understanding the algorithmic structure behind quantum chemistry programs. It currently focuses on small molecular systems and educational clarity rather than production-level performance.

## Current features

Implemented methods and modules:

- Gaussian basis-set integral evaluation through **Libint2**
- XYZ molecular geometry input
- Restricted Hartree-Fock, RHF
- Symmetric orthogonalization, \(S^{-1/2}\)
- Direct shell-block Fock build
- DIIS acceleration for SCF convergence
- SCF warm start using previous molecular orbital coefficients
- Numerical-gradient geometry optimization
- Cartesian BFGS optimizer
- MP2 single-point energy
  - dense AO/MO prototype
  - OVOV-only transformation
  - direct first-index transformation without storing the full AO ERI tensor
- FCI, full configuration interaction
- CID, configuration interaction with doubles
- CISD, configuration interaction with singles and doubles

The project has been validated against PySCF for several small systems.

## Dependencies

Required:

- C++17 compiler
- Eigen3
- Libint2
- OpenMP, optional but recommended

A conda-based environment is convenient:

```bash
conda create -n abinit-study -c conda-forge \
  cxx-compiler eigen libint cmake ninja -y

conda activate abinit-study
```

If Libint basis files are not found automatically, set `LIBINT_DATA_PATH`, for example:

```bash
export LIBINT_DATA_PATH=$CONDA_PREFIX/share/libint
```

Depending on the Libint package layout, this path may need to point to the parent directory that contains the `basis` folder.

## Build

A simple direct compilation command is:

```bash
g++ -std=c++17 -O2 -fopenmp \
  mini_rhf_libint_xyz_diis_opt_mp2_fci.cpp \
  geometry_optimizer.cpp \
  mp2_v3_direct_t1.cpp \
  fci.cpp \
  -o RHF \
  -I. \
  -I$CONDA_PREFIX/include \
  -I$CONDA_PREFIX/include/eigen3 \
  -L$CONDA_PREFIX/lib \
  -Wl,-rpath,$CONDA_PREFIX/lib \
  -lint2
```

The exact source filenames may change as the project develops, but the typical build structure is:

```text
main program
+ geometry optimizer module
+ MP2 module
+ CI/FCI module
+ Libint2
+ Eigen3
```

## Basic usage

General command format:

```bash
./RHF BASIS_NAME molecule.xyz UNIT CHARGE MULTIPLICITY [options]
```

Example:

```bash
./RHF sto-3g h2_bohr.xyz bohr 0 1
```

For a standard XYZ file in Angstrom:

```bash
./RHF 6-31g water_angstrom.xyz angstrom 0 1
```

### RHF single-point calculation

```bash
./RHF sto-3g h2_bohr.xyz bohr 0 1
```

### MP2 single-point calculation

```bash
./RHF cc-pvqz water_angstrom.xyz angstrom 0 1 --mp2
```

The current MP2 implementation uses a direct first-index transformation:

\[
t_1(i,\nu,\lambda,\sigma)
=
\sum_\mu C_{\mu i}(\mu\nu|\lambda\sigma)
\]

so the full AO ERI tensor is not stored in the newest MP2 version.

### Geometry optimization

```bash
./RHF sto-3g water_angstrom.xyz angstrom 0 1 --opt --opt-out water_opt.xyz
```

The geometry optimizer currently uses:

- central finite-difference numerical gradients
- Cartesian coordinates
- BFGS inverse-Hessian update
- Armijo backtracking line search
- optional SCF warm start from previous MO coefficients

### FCI

```bash
./RHF sto-3g h2_bohr.xyz bohr 0 1 --fci
```

To print the largest CI coefficients:

```bash
./RHF sto-3g h2_bohr.xyz bohr 0 1 --fci --fci-coeffs
```

### CID

```bash
./RHF 6-31g water_angstrom.xyz angstrom 0 1 --cid
```

CID includes the Hartree-Fock reference determinant and all double excitations.

### CISD

```bash
./RHF 6-31g water_angstrom.xyz angstrom 0 1 --cisd
```

CISD includes the Hartree-Fock reference determinant, all single excitations, and all double excitations.

## Implemented theory

### RHF

The RHF module solves the Roothaan equation:

\[
FC = SC\epsilon
\]

using symmetric orthogonalization:

\[
X = S^{-1/2}
\]

\[
F' = X^T F X
\]

The Fock matrix is constructed as:

\[
F_{\mu\nu}
=
H_{\mu\nu}^{core}
+
G_{\mu\nu}
\]

with

\[
G_{\mu\nu}
=
\sum_{\lambda\sigma}
D_{\lambda\sigma}
\left[
(\mu\nu|\lambda\sigma)
-
\frac{1}{2}
(\mu\lambda|\nu\sigma)
\right]
\]

### DIIS

The DIIS error matrix is based on the generalized commutator:

\[
e = FDS - SDF
\]

and is transformed to the orthogonalized AO basis:

\[
e' = X^T(FDS-SDF)X
\]

The Fock matrix is extrapolated from previous Fock/error pairs.

### MP2

The closed-shell canonical RHF-MP2 correlation energy is computed as:

\[
E_{\mathrm{MP2}}^{(2)}
=
\sum_{ij}^{occ}
\sum_{ab}^{vir}
\frac{
(ia|jb)
\left[
2(ia|jb)-(ib|ja)
\right]
}{
\epsilon_i+\epsilon_j-\epsilon_a-\epsilon_b
}
\]

The MP2 total energy is:

\[
E_{\mathrm{MP2}}
=
E_{\mathrm{HF}}
+
E_{\mathrm{MP2}}^{(2)}
\]

### FCI

FCI is performed in a determinant basis with fixed \(N_\alpha\) and \(N_\beta\). Determinants are represented as bitstrings.

For closed-shell singlet systems:

\[
N_\alpha = N_\beta = N_e/2
\]

The FCI wavefunction is:

\[
|\Psi_{\mathrm{FCI}}\rangle
=
\sum_I c_I |\Phi_I\rangle
\]

and the CI Hamiltonian is diagonalized:

\[
\mathbf H \mathbf c = E \mathbf c
\]

### CID and CISD

CID wavefunction:

\[
|\Psi_{\mathrm{CID}}\rangle
=
c_0|\Phi_0\rangle
+
\sum_{ijab}c_{ij}^{ab}|\Phi_{ij}^{ab}\rangle
\]

CISD wavefunction:

\[
|\Psi_{\mathrm{CISD}}\rangle
=
c_0|\Phi_0\rangle
+
\sum_{ia}c_i^a|\Phi_i^a\rangle
+
\sum_{ijab}c_{ij}^{ab}|\Phi_{ij}^{ab}\rangle
\]

The determinant space is generated directly from the RHF reference determinant by bit flips, rather than by generating the full FCI space and filtering.

## Example validation

### H2/STO-3G RHF

For H2 at \(R = 1.4\) bohr:

```text
E_RHF approx -1.1167 Ha
```

### H2O/6-31G CISD

The CISD energy has been checked against PySCF and agrees to approximately \(10^{-9}\) Ha in tested cases.

Example PySCF comparison:

```python
from pyscf import gto, scf, ci

mol = gto.M(
    atom='''
    O 0.000000 0.000000 0.000000
    H 0.000000 0.757160 0.586260
    H 0.000000 -0.757160 0.586260
    ''',
    unit='Angstrom',
    basis='6-31g',
    charge=0,
    spin=0,
)

mf = scf.RHF(mol).run()

myci = ci.CISD(mf)
ecorr, civec = myci.kernel()

print("E_HF =", mf.e_tot)
print("E_CISD_corr =", ecorr)
print("E_CISD_total =", mf.e_tot + ecorr)
```

## Notes on determinant representation

The CI module represents a determinant as a 64-bit integer:

```text
lower nmo bits: alpha occupation
upper nmo bits: beta occupation
```

For example:

```text
det = alpha_bits | (beta_bits << nmo)
```

The fermionic phase factors are handled by creation and annihilation operators using `popcount`.

The bitwise XOR operator is used to generate excitations:

```cpp
new_bits = ref_bits ^ (1ULL << i) ^ (1ULL << a);
```

This clears an occupied orbital \(i\) and sets a virtual orbital \(a\).

## Limitations

This project is educational and intentionally simple.

Current limitations include:

- closed-shell RHF focus
- no unrestricted HF yet
- no analytic gradients yet
- no ECP/pseudopotential support
- no point-group symmetry
- no spin-adapted CI basis
- dense CI Hamiltonian construction
- dense diagonalization through Eigen3
- no Davidson/Lanczos CI solver yet
- no RI/DF approximation yet
- no DFT module yet
- no periodic boundary conditions yet

For FCI, determinant-space growth is combinatorial:

\[
N_{\mathrm{det}}
=
\binom{n_{\mathrm{mo}}}{N_\alpha}
\binom{n_{\mathrm{mo}}}{N_\beta}
\]

Therefore FCI is only suitable for very small systems and small basis sets in the current implementation.

## Suggested development roadmap

Near-term:

- RKS-LDA DFT implementation
- numerical integration grid
- AO values on grid
- electron density \(\rho(\mathbf r)\)
- LDA exchange-correlation energy
- \(V_{xc}\) matrix construction

Later:

- GGA/PBE through Libxc or self-written prototype
- UHF
- analytic RHF gradient
- Davidson diagonalization for CI
- active-space CI / CASCI
- Schwarz screening
- blocked AO-to-MO transformation
- RI-MP2
- more systematic performance optimization

## Purpose

The goal of this repository is not to replace mature quantum chemistry packages such as PySCF, Gaussian, ORCA, Q-Chem, CP2K, or Psi4.

The goal is to learn how electronic-structure methods are translated into working C++ code:

```text
mathematical formula
-> data structure
-> tensor/index convention
-> numerical algorithm
-> validated implementation
```

This project is mainly for studying the internal logic of ab initio quantum chemistry methods.
