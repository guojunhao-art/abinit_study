# abinit_study

A small C++ learning project for implementing basic **ab initio electronic-structure methods** from scratch.

The repository is intended for understanding how quantum-chemistry formulas are translated into concrete data structures, tensor-index conventions, and numerical algorithms.  It currently targets small molecular systems and educational clarity rather than production-level performance.

## Current status

The code currently supports small closed-shell molecular calculations with Gaussian basis functions through Libint2.  The most mature path is closed-shell RHF plus analytic-gradient geometry optimization.

Implemented or partially implemented features:

- Molecular XYZ input in Angstrom or bohr.
- Gaussian basis-set integral evaluation through **Libint2**.
- Closed-shell restricted Hartree-Fock, RHF.
- Symmetric orthogonalization using the overlap matrix.
- Direct shell-block Coulomb/exchange Fock builds.
- Pulay DIIS acceleration for SCF convergence.
- SCF warm start from previous molecular orbital coefficients.
- RHF analytic nuclear gradient.
- Cartesian BFGS geometry optimization using the RHF analytic gradient.
- Optional finite-difference checks for integral derivatives and RHF gradients.
- RHF-MP2 single-point energy.
  - older dense AO/MO prototype.
  - OVOV-only transformation.
  - direct first-index transformation without storing the full AO ERI tensor.
- Determinant CI methods:
  - FCI.
  - CID.
  - CISD.
- Experimental DFT/RKS infrastructure:
  - atom-centered numerical integration grids.
  - radial grid plus small Lebedev-like angular grids.
  - distance or Becke partitioning.
  - AO values and density on grids.
  - Libxc-backed LDA/GGA hooks, including a PBE path.
- Ongoing staged refactor:
  - shared molecule, XYZ, linear-algebra, one-body integral, DIIS, Fock-build, and RHF helper headers.
  - `miniqc.hpp` aggregate header for new small drivers/tests.

The code has been compared against PySCF for several small RHF, MP2, and CI test cases.  The analytic RHF gradient has been checked against central finite-difference gradients.

## Dependencies

Required:

- C++17 compiler.
- Eigen3.
- Libint2.
- Libxc, for the DFT/RKS path.
- OpenMP, optional but recommended.

A conda-based environment is convenient:

```bash
conda create -n abinit-study -c conda-forge \
  cxx-compiler eigen libint libxc cmake ninja -y

conda activate abinit-study
```

If Libint basis files are not found automatically, set `LIBINT_DATA_PATH`, for example:

```bash
export LIBINT_DATA_PATH=$CONDA_PREFIX/share/libint
```

Depending on the Libint package layout, this path may need to point to the parent directory that contains the `basis` folder.

## Build

At this stage the project is still usually built by a direct compile command.  A typical command is:

```bash
g++ -std=c++17 -O2 -fopenmp \
  main.cpp \
  geometry_optimizer.cpp \
  analytic_gradient.cpp \
  mp2.cpp \
  fci.cpp \
  dft_grid.cpp \
  dft_lda.cpp \
  -o RHF \
  -I. \
  -I$CONDA_PREFIX/include \
  -I$CONDA_PREFIX/include/eigen3 \
  -L$CONDA_PREFIX/lib \
  -Wl,-rpath,$CONDA_PREFIX/lib \
  -lint2 -lxc
```

The current refactor is moving the code toward a cleaner library-plus-driver structure, but the historical executable is still `main.cpp`.

## Basic usage

General command format:

```bash
./RHF BASIS_NAME molecule.xyz [unit=angstrom] [charge=0] [multiplicity=1] [options]
```

Examples:

```bash
# H2 coordinates already in bohr
./RHF sto-3g h2_bohr.xyz bohr 0 1

# Standard XYZ coordinates in Angstrom
./RHF 6-31g water.xyz angstrom 0 1
```

Available options in the current driver include:

```text
--print-matrices          print one-electron matrices and final density/MO matrices
--mp2                     run RHF-MP2 after RHF
--fci                     run full CI after RHF
--cid                     run CI with doubles after RHF
--cisd                    run CI with singles and doubles after RHF
--coeffs                  print largest CI coefficients; use with --fci/--cid/--cisd
--opt                     run RHF analytic-gradient geometry optimization
--opt-out file.xyz        output optimized geometry; default optimized.xyz
--lda-x nrad nang         run the experimental RKS/DFT grid path
```

Only one of `--fci`, `--cid`, and `--cisd` should be used at a time.

### RHF single point

```bash
./RHF sto-3g h2_bohr.xyz bohr 0 1
```

### RHF analytic-gradient geometry optimization

```bash
./RHF 6-311g** water.xyz angstrom 0 1 --opt --opt-out water_opt.xyz
```

The optimizer currently uses Cartesian coordinates, an inverse-Hessian BFGS update, Armijo backtracking, analytic RHF gradients, and optional SCF warm starts through previous MO coefficients.

### MP2 single point

```bash
./RHF cc-pvdz water.xyz angstrom 0 1 --mp2
```

The current direct MP2 implementation transforms the AO ERI tensor one index at a time, so it avoids storing the full four-index AO tensor in the newest path.

### FCI, CID, and CISD

```bash
./RHF sto-3g h2_bohr.xyz bohr 0 1 --fci
./RHF 6-31g water.xyz angstrom 0 1 --cid
./RHF 6-31g water.xyz angstrom 0 1 --cisd --coeffs
```

The determinant CI module uses bitstrings for alpha and beta occupations and dense Hamiltonian diagonalization through Eigen.

### Experimental RKS/DFT path

```bash
./RHF 6-31g water.xyz angstrom 0 1 --lda-x 80 26
```

Despite the historical option name `--lda-x`, the DFT code now contains Libxc-backed LDA/GGA hooks.  This path should still be treated as experimental; the RHF/MP2/CI/gradient path is more mature.

## Code organization after the current refactor

Historical files still used by the executable:

```text
main.cpp
geometry_optimizer.hpp/.cpp
analytic_gradient.hpp/.cpp
mp2_v3_direct_t1.hpp + mp2.cpp
fci.hpp/.cpp
cisd.hpp
dft_grid.hpp/.cpp
dft_lda.hpp/.cpp
lebedev_grid.hpp
```

Shared helper headers introduced during the staged refactor:

```text
core_molecule.hpp       Molecule, electron count, nuclear repulsion, coordinate vector helpers
periodic_table.hpp      element symbol and atomic-number helpers
xyz_io.hpp              XYZ read/write helpers
linalg_utils.hpp        orthogonalization, density, trace, Fock diagonalization
one_body_integrals.hpp  overlap, kinetic, nuclear attraction, Hcore builders
scf_diis.hpp            shared Pulay DIIS helper
basis_context.hpp       centralized Libint basis construction
two_body_fock.hpp       direct J and RHF G builders
rhf.hpp                 reusable RHF options/result/driver wrappers
miniqc.hpp              aggregate include header for new small drivers/tests
```

The next structural goal is to shrink `main.cpp` into a thin command-line driver and move calculation workflows into reusable modules.

## Theory notes

This section is intentionally written as a compact review reference for later study.

### AO basis and one-electron integrals

Molecular orbitals are expanded in atom-centered AO basis functions:

$$
\psi_p(\mathbf r) = \sum_\mu C_{\mu p}\chi_\mu(\mathbf r).
$$

The overlap matrix is

$$
S_{\mu\nu}=\langle \chi_\mu | \chi_\nu \rangle.
$$

The one-electron core Hamiltonian is

$$
H^{\mathrm{core}}_{\mu\nu}=T_{\mu\nu}+V_{\mu\nu},
$$

with

$$
T_{\mu\nu}=\left\langle \chi_\mu \left| -\frac{1}{2}\nabla^2 \right| \chi_\nu \right\rangle,
$$

and

$$
V_{\mu\nu}=\left\langle \chi_\mu \left| -\sum_A \frac{Z_A}{|\mathbf r-\mathbf R_A|} \right| \chi_\nu \right\rangle.
$$

The two-electron repulsion integral is written as

$$
(\mu\nu|\lambda\sigma)
=
\iint
\chi_\mu(\mathbf r_1)\chi_\nu(\mathbf r_1)
\frac{1}{r_{12}}
\chi_\lambda(\mathbf r_2)\chi_\sigma(\mathbf r_2)
\,d\mathbf r_1\,d\mathbf r_2.
$$

### Closed-shell RHF

For closed-shell RHF, each occupied spatial orbital contains two electrons.  The spin-summed AO density matrix is

$$
D_{\mu\nu}=2\sum_i^{\mathrm{occ}} C_{\mu i}C_{\nu i}.
$$

The RHF Fock matrix is

$$
F_{\mu\nu}=H^{\mathrm{core}}_{\mu\nu}+G_{\mu\nu},
$$

where

$$
G_{\mu\nu}
=
\sum_{\lambda\sigma}D_{\lambda\sigma}
\left[
(\mu\nu|\lambda\sigma)
-
\frac{1}{2}(\mu\lambda|\nu\sigma)
\right].
$$

The electronic RHF energy is

$$
E_{\mathrm{elec}}
=
\frac{1}{2}\sum_{\mu\nu}D_{\mu\nu}
\left(H^{\mathrm{core}}_{\mu\nu}+F_{\mu\nu}\right),
$$

and the total energy is

$$
E_{\mathrm{RHF}}=E_{\mathrm{elec}}+E_{\mathrm{nuc}},
$$

where

$$
E_{\mathrm{nuc}}=\sum_{A<B}\frac{Z_A Z_B}{R_{AB}}.
$$

The Roothaan equation is the generalized eigenvalue problem

$$
FC=SC\varepsilon.
$$

The code solves it by symmetric orthogonalization.  If

$$
X=S^{-1/2},
$$

then

$$
F'=X^T F X,
$$

and the standard eigenproblem is

$$
F'C'=C'\varepsilon,
\qquad
C=XC'.
$$

### DIIS

At SCF convergence, the density and Fock matrices satisfy the generalized commutator condition

$$
FDS-SDF=0.
$$

The implementation transforms this error into the orthogonalized AO basis:

$$
e_i=X^T(F_iD_iS-SD_iF_i)X.
$$

DIIS forms an extrapolated Fock matrix

$$
F^{\mathrm{DIIS}}=\sum_i c_i F_i
$$

by minimizing the norm of the extrapolated error subject to the constraint

$$
\sum_i c_i=1.
$$

This gives the Pulay linear system

$$
\begin{pmatrix}
B & -\mathbf 1\\
-\mathbf 1^T & 0
\end{pmatrix}
\begin{pmatrix}
\mathbf c\\
\lambda
\end{pmatrix}
=
\begin{pmatrix}
\mathbf 0\\
-1
\end{pmatrix},
\qquad
B_{ij}=\langle e_i,e_j\rangle.
$$

### RHF analytic gradient

For nuclear coordinate \(R_A^x\), the RHF gradient can be written schematically as

$$
\frac{dE}{dR_A^x}
=
\frac{dE_{\mathrm{nuc}}}{dR_A^x}
+
\sum_{\mu\nu}D_{\mu\nu}\frac{\partial H^{\mathrm{core}}_{\mu\nu}}{\partial R_A^x}
+
\frac{1}{2}\sum_{\mu\nu\lambda\sigma}
D_{\mu\nu}D_{\lambda\sigma}
\frac{\partial}{\partial R_A^x}
\left[
(\mu\nu|\lambda\sigma)
-
\frac{1}{2}(\mu\lambda|\nu\sigma)
\right]
-
\sum_{\mu\nu}W_{\mu\nu}
\frac{\partial S_{\mu\nu}}{\partial R_A^x}.
$$

The last term is the Pulay overlap term.  In a canonical closed-shell RHF basis, the energy-weighted density matrix can be written as

$$
W_{\mu\nu}
=
2\sum_i^{\mathrm{occ}} \varepsilon_i C_{\mu i}C_{\nu i}.
$$

The force reported by the program is the negative gradient:

$$
F_A^x=-\frac{dE}{dR_A^x}.
$$

The implementation currently evaluates derivative overlap, kinetic, nuclear-attraction, and two-electron integral contributions and compares them against finite differences during validation.

### BFGS geometry optimization

Geometry optimization minimizes the total RHF energy as a function of Cartesian nuclear coordinates:

$$
E=E(\mathbf x),
\qquad
\mathbf x=(x_1,y_1,z_1,x_2,y_2,z_2,\ldots).
$$

BFGS approximates the inverse Hessian \(H_k\).  With

$$
\mathbf s_k=\mathbf x_{k+1}-\mathbf x_k,
\qquad
\mathbf y_k=\mathbf g_{k+1}-\mathbf g_k,
$$

and

$$
\rho_k=\frac{1}{\mathbf y_k^T\mathbf s_k},
$$

the inverse-Hessian update is

$$
H_{k+1}
=
(I-\rho_k\mathbf s_k\mathbf y_k^T)H_k
(I-\rho_k\mathbf y_k\mathbf s_k^T)
+
\rho_k\mathbf s_k\mathbf s_k^T.
$$

The search direction is

$$
\mathbf p_k=-H_k\mathbf g_k.
$$

The current implementation uses Armijo backtracking and analytic RHF gradients.

### MP2

For canonical closed-shell RHF orbitals, the MP2 correlation energy is

$$
E_{\mathrm{MP2}}^{(2)}
=
\sum_{ij}^{\mathrm{occ}}\sum_{ab}^{\mathrm{vir}}
\frac{(ia|jb)\left[2(ia|jb)-(ib|ja)\right]}
{\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b}.
$$

The total MP2 energy is

$$
E_{\mathrm{MP2}}=E_{\mathrm{RHF}}+E_{\mathrm{MP2}}^{(2)}.
$$

The direct first-index transformation used in the newer MP2 path starts from

$$
t_1(i,\nu,\lambda,\sigma)
=
\sum_\mu C_{\mu i}(\mu\nu|\lambda\sigma),
$$

then continues transforming only the blocks needed for occupied-virtual combinations.

### Determinant CI

The CI wavefunction is expanded in Slater determinants:

$$
|\Psi\rangle=\sum_I c_I|\Phi_I\rangle.
$$

The CI coefficients are obtained from

$$
\mathbf H\mathbf c=E\mathbf c.
$$

For a closed-shell singlet reference,

$$
N_\alpha=N_\beta=N_e/2.
$$

FCI includes all determinants with the target \(N_\alpha\) and \(N_\beta\).  CID includes the reference plus double excitations:

$$
|\Psi_{\mathrm{CID}}\rangle
=
c_0|\Phi_0\rangle+
\sum_{ijab}c_{ij}^{ab}|\Phi_{ij}^{ab}\rangle.
$$

CISD includes the reference, singles, and doubles:

$$
|\Psi_{\mathrm{CISD}}\rangle
=
c_0|\Phi_0\rangle+
\sum_{ia}c_i^a|\Phi_i^a\rangle+
\sum_{ijab}c_{ij}^{ab}|\Phi_{ij}^{ab}\rangle.
$$

The determinant count for FCI scales combinatorially:

$$
N_{\mathrm{det}}
=
\binom{n_{\mathrm{mo}}}{N_\alpha}
\binom{n_{\mathrm{mo}}}{N_\beta}.
$$

In this code, determinants are represented with 64-bit occupation bitstrings:

```text
lower nmo bits: alpha occupation
upper nmo bits: beta occupation
```

A determinant is encoded as

```text
det = alpha_bits | (beta_bits << nmo)
```

Creation and annihilation phase factors are handled through bit operations and population counts.

### Numerical DFT/RKS notes

The grid code evaluates integrals of the form

$$
\int f(\mathbf r)\,d\mathbf r
\approx
\sum_g w_g f(\mathbf r_g).
$$

For a closed-shell density,

$$
\rho(\mathbf r_g)=\sum_{\mu\nu}D_{\mu\nu}\chi_\mu(\mathbf r_g)\chi_\nu(\mathbf r_g).
$$

The exchange-correlation energy is approximated as

$$
E_{xc}\approx\sum_g w_g\,\rho(\mathbf r_g)\,\varepsilon_{xc}(\rho,\nabla\rho,\ldots).
$$

For a local functional, the XC matrix has the schematic form

$$
V^{xc}_{\mu\nu}
\approx
\sum_g w_g\,v_{xc}(\mathbf r_g)\chi_\mu(\mathbf r_g)\chi_\nu(\mathbf r_g),
$$

with additional gradient-dependent terms for GGA functionals.

## Example validation

### H2/STO-3G RHF

For H2 near \(R=1.4\) bohr, the RHF energy is approximately

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

### H2O/6-311G** RHF geometry optimization

The current analytic-gradient BFGS path has been tested on water with a 6-311G** basis.  The analytic-gradient optimization gives the same optimized geometry as the older finite-difference-gradient workflow, while reducing the runtime from tens of seconds to roughly a few seconds on the tested machine.

## Limitations

This project is educational and intentionally simple.  Current limitations include:

- Closed-shell RHF/RKS focus; no UHF/UKS yet.
- No analytic Hessian or vibrational analysis.
- RHF analytic gradients are implemented, but MP2/CI/DFT gradients are not.
- No ECP/pseudopotential support.
- No point-group symmetry.
- No spin-adapted CI basis.
- Dense CI Hamiltonian construction.
- Dense diagonalization through Eigen3.
- No Davidson/Lanczos CI solver yet.
- No RI/DF approximation yet.
- DFT/RKS path is experimental and grid-sensitive.
- No periodic boundary conditions yet.

## Suggested development roadmap

Near-term engineering tasks:

- Shrink `main.cpp` into a thin command-line driver.
- Move single-point RHF, optimization, MP2, CI, and DFT workflows into separate driver modules.
- Add CMake and small regression tests.
- Disable finite-difference derivative checks by default and expose them through explicit flags.
- Centralize basis construction through `BasisContext` everywhere.

Near-term method tasks:

- Make RKS/LDA/PBE path more systematic and validate against PySCF.
- Add cleaner grid convergence tests.
- Add Davidson diagonalization for CI.
- Add Schwarz screening for direct Fock builds.
- Add active-space CI / CASCI.

Later tasks:

- UHF/UKS.
- MP2 gradient.
- RI-MP2.
- More systematic AO-to-MO transformation blocking.
- Periodic boundary conditions.

## Purpose

The goal of this repository is not to replace mature quantum-chemistry packages such as PySCF, Gaussian, ORCA, Q-Chem, CP2K, or Psi4.

The goal is to learn how electronic-structure methods are translated into working C++ code:

```text
mathematical formula
-> data structure
-> tensor/index convention
-> numerical algorithm
-> validated implementation
```

This project is mainly for studying the internal logic of ab initio quantum chemistry methods.
