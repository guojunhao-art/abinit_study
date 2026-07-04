# abinit_study

A small C++ learning project for implementing basic **ab initio electronic-structure methods** from scratch.

The goal is not to replace PySCF, Gaussian, ORCA, Q-Chem, CP2K, or Psi4.  The goal is to learn how electronic-structure formulas become working C++ code:

```text
mathematical formula
-> data structure
-> tensor/index convention
-> numerical algorithm
-> validated implementation
```

The current code targets small molecular systems and educational clarity rather than production-level performance.

## Current features

Implemented or partially implemented features:

- XYZ molecular geometry input in Angstrom or bohr.
- Gaussian basis-set integral evaluation through **Libint2**.
- Closed-shell restricted Hartree-Fock, RHF.
- Symmetric orthogonalization, direct shell-block Fock builds, and Pulay DIIS.
- SCF warm start using previous molecular orbital coefficients.
- RHF analytic nuclear gradient.
- Cartesian BFGS geometry optimization using either:
  - RHF analytic gradients, or
  - central finite-difference numerical gradients.
- RHF-MP2 single-point energy.
- Determinant CI methods:
  - FCI.
  - CID.
  - CISD.
- Experimental RKS/DFT infrastructure:
  - atom-centered numerical integration grids.
  - distance or Becke partitioning.
  - AO values and density on grids.
  - Libxc-backed LDA/GGA hooks, including a PBE path.
- Staged refactor toward reusable modules:
  - molecule/XYZ helpers.
  - linear-algebra helpers.
  - one-electron and two-electron integral helpers.
  - shared DIIS helper.
  - reusable RHF driver.
  - config-file driver.

The RHF/MP2/CI paths have been compared against PySCF for small systems.  The RHF analytic gradient has been checked against finite-difference gradients.

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

The project is still usually built by a direct compile command:

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

The next engineering step is to add CMake and split the code into a library plus small executables/tests.

## Configuration-file driver

The driver now reads an INI-style configuration file instead of requiring a long positional command line.

Usage:

```bash
./RHF input.in
./RHF --config input.in
./RHF --write-template-config > input.in
```

A minimal RHF single-point input is:

```ini
[molecule]
basis = 6-31g
xyz = water.xyz
unit = angstrom
charge = 0
multiplicity = 1

[calculation]
job = single_point
mp2 = false
ci = none
dft = false
```

A geometry optimization input using the RHF analytic gradient is:

```ini
[molecule]
basis = 6-311g**
xyz = water.xyz
unit = angstrom
charge = 0
multiplicity = 1

[calculation]
job = optimize

[geometry]
gradient = analytic
output = water_opt.xyz
max_steps = 50
max_step = 0.20
```

To deliberately use the older numerical-gradient path, change only this line:

```ini
[geometry]
gradient = numerical
```

Post-HF options are configured as:

```ini
[calculation]
job = single_point
mp2 = true
ci = cisd     # none, fci, cid, or cisd
dft = false

[ci]
print_coefficients = true
max_determinants = 5000
max_nmo = 16
max_tensor_mb = 4096.0
```

The experimental DFT/RKS path can be enabled with:

```ini
[calculation]
dft = true

[dft]
functional = pbe       # slater_x, lda_x, lda_x_pz81, or pbe
n_radial = 80
angular_grid = 26
density_mixing = 0.3
```

## Code organization

Historical method modules:

```text
geometry_optimizer.hpp/.cpp
analytic_gradient.hpp/.cpp
mp2_v3_direct_t1.hpp + mp2.cpp
fci.hpp/.cpp
cisd.hpp
dft_grid.hpp/.cpp
dft_lda.hpp/.cpp
lebedev_grid.hpp
```

Shared helper headers introduced during the refactor:

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
calculation_config.hpp  INI-style config-file parser and driver options
miniqc.hpp              aggregate include header for core/RHF helpers
```

`main.cpp` is now intended to be a thin config-file driver rather than a place for integral tests, finite-difference derivative checks, and method implementations.

## Theory notes

This section is a compact review note for later study.  The formulas are written in fenced `math` blocks because they are more robust across GitHub Markdown renderers than multi-line `$$...$$` blocks.

### AO basis and one-electron integrals

Molecular orbitals are expanded in atom-centered AO basis functions:

```math
\psi_p(\mathbf r) = \sum_\mu C_{\mu p}\chi_\mu(\mathbf r).
```

The overlap matrix is

```math
S_{\mu\nu}=\langle \chi_\mu | \chi_\nu \rangle.
```

The one-electron core Hamiltonian is

```math
H^{\mathrm{core}}_{\mu\nu}=T_{\mu\nu}+V_{\mu\nu}.
```

The kinetic-energy integral is

```math
T_{\mu\nu}=\left\langle \chi_\mu \left| -\frac{1}{2}\nabla^2 \right| \chi_\nu \right\rangle.
```

The electron-nuclear attraction integral is

```math
V_{\mu\nu}=\left\langle \chi_\mu \left| -\sum_A \frac{Z_A}{|\mathbf r-\mathbf R_A|} \right| \chi_\nu \right\rangle.
```

The two-electron repulsion integral is

```math
(\mu\nu|\lambda\sigma)=\iint \chi_\mu(\mathbf r_1)\chi_\nu(\mathbf r_1)\frac{1}{r_{12}}\chi_\lambda(\mathbf r_2)\chi_\sigma(\mathbf r_2)\,d\mathbf r_1\,d\mathbf r_2.
```

### Closed-shell RHF

For closed-shell RHF, the spin-summed AO density matrix is

```math
D_{\mu\nu}=2\sum_i^{\mathrm{occ}} C_{\mu i}C_{\nu i}.
```

The RHF Fock matrix is

```math
F_{\mu\nu}=H^{\mathrm{core}}_{\mu\nu}+G_{\mu\nu}.
```

The two-electron contribution is

```math
G_{\mu\nu}=\sum_{\lambda\sigma}D_{\lambda\sigma}\left[(\mu\nu|\lambda\sigma)-\frac{1}{2}(\mu\lambda|\nu\sigma)\right].
```

The electronic RHF energy is

```math
E_{\mathrm{elec}}=\frac{1}{2}\sum_{\mu\nu}D_{\mu\nu}\left(H^{\mathrm{core}}_{\mu\nu}+F_{\mu\nu}\right).
```

The total RHF energy is

```math
E_{\mathrm{RHF}}=E_{\mathrm{elec}}+E_{\mathrm{nuc}},\qquad E_{\mathrm{nuc}}=\sum_{A<B}\frac{Z_AZ_B}{R_{AB}}.
```

The Roothaan equation is the generalized eigenvalue problem

```math
FC=SC\varepsilon.
```

Using symmetric orthogonalization,

```math
X=S^{-1/2},\qquad F'=X^T F X,\qquad F'C'=C'\varepsilon,\qquad C=XC'.
```

### DIIS

At SCF convergence, the density and Fock matrices satisfy

```math
FDS-SDF=0.
```

The implementation transforms this error into the orthogonalized AO basis:

```math
e_i=X^T(F_iD_iS-SD_iF_i)X.
```

DIIS forms an extrapolated Fock matrix

```math
F^{\mathrm{DIIS}}=\sum_i c_iF_i,\qquad \sum_i c_i=1.
```

The coefficients are obtained from the Pulay linear system:

```math
\begin{pmatrix}B&-\mathbf 1\\-\mathbf 1^T&0\end{pmatrix}\begin{pmatrix}\mathbf c\\\lambda\end{pmatrix}=\begin{pmatrix}\mathbf 0\\-1\end{pmatrix},\qquad B_{ij}=\langle e_i,e_j\rangle.
```

### RHF analytic gradient

For a nuclear coordinate \(R_A^x\), the RHF gradient can be written schematically as

```math
\frac{dE}{dR_A^x}=\frac{dE_{\mathrm{nuc}}}{dR_A^x}+\sum_{\mu\nu}D_{\mu\nu}\frac{\partial H^{\mathrm{core}}_{\mu\nu}}{\partial R_A^x}+\frac{1}{2}\sum_{\mu\nu\lambda\sigma}D_{\mu\nu}D_{\lambda\sigma}\frac{\partial}{\partial R_A^x}\left[(\mu\nu|\lambda\sigma)-\frac{1}{2}(\mu\lambda|\nu\sigma)\right]-\sum_{\mu\nu}W_{\mu\nu}\frac{\partial S_{\mu\nu}}{\partial R_A^x}.
```

The last term is the Pulay overlap term.  For canonical closed-shell RHF orbitals, the energy-weighted density is

```math
W_{\mu\nu}=2\sum_i^{\mathrm{occ}}\varepsilon_i C_{\mu i}C_{\nu i}.
```

The force is the negative gradient:

```math
F_A^x=-\frac{dE}{dR_A^x}.
```

### BFGS geometry optimization

The geometry optimizer minimizes the RHF total energy in Cartesian nuclear coordinates:

```math
E=E(\mathbf x),\qquad \mathbf x=(x_1,y_1,z_1,x_2,y_2,z_2,\ldots).
```

With

```math
\mathbf s_k=\mathbf x_{k+1}-\mathbf x_k,\qquad \mathbf y_k=\mathbf g_{k+1}-\mathbf g_k,\qquad \rho_k=\frac{1}{\mathbf y_k^T\mathbf s_k},
```

the inverse-Hessian BFGS update is

```math
H_{k+1}=(I-\rho_k\mathbf s_k\mathbf y_k^T)H_k(I-\rho_k\mathbf y_k\mathbf s_k^T)+\rho_k\mathbf s_k\mathbf s_k^T.
```

The search direction is

```math
\mathbf p_k=-H_k\mathbf g_k.
```

### MP2

For canonical closed-shell RHF orbitals, the MP2 correlation energy is

```math
E_{\mathrm{MP2}}^{(2)}=\sum_{ij}^{\mathrm{occ}}\sum_{ab}^{\mathrm{vir}}\frac{(ia|jb)\left[2(ia|jb)-(ib|ja)\right]}{\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b}.
```

The total MP2 energy is

```math
E_{\mathrm{MP2}}=E_{\mathrm{RHF}}+E_{\mathrm{MP2}}^{(2)}.
```

The newer direct MP2 path begins with a first-index AO-to-MO transformation:

```math
t_1(i,\nu,\lambda,\sigma)=\sum_\mu C_{\mu i}(\mu\nu|\lambda\sigma).
```

### Determinant CI

The CI wavefunction is expanded in Slater determinants:

```math
|\Psi\rangle=\sum_I c_I|\Phi_I\rangle.
```

The CI coefficients are obtained from

```math
\mathbf H\mathbf c=E\mathbf c.
```

FCI includes all determinants with the target \(N_\alpha\) and \(N_\beta\).  For a closed-shell singlet,

```math
N_\alpha=N_\beta=N_e/2.
```

CID includes the reference plus double excitations:

```math
|\Psi_{\mathrm{CID}}\rangle=c_0|\Phi_0\rangle+\sum_{ijab}c_{ij}^{ab}|\Phi_{ij}^{ab}\rangle.
```

CISD includes the reference, singles, and doubles:

```math
|\Psi_{\mathrm{CISD}}\rangle=c_0|\Phi_0\rangle+\sum_{ia}c_i^a|\Phi_i^a\rangle+\sum_{ijab}c_{ij}^{ab}|\Phi_{ij}^{ab}\rangle.
```

The FCI determinant count scales as

```math
N_{\mathrm{det}}=\binom{n_{\mathrm{mo}}}{N_\alpha}\binom{n_{\mathrm{mo}}}{N_\beta}.
```

The code represents a determinant as a 64-bit integer:

```text
lower nmo bits: alpha occupation
upper nmo bits: beta occupation

det = alpha_bits | (beta_bits << nmo)
```

### Numerical DFT/RKS notes

The grid code approximates real-space integrals as

```math
\int f(\mathbf r)\,d\mathbf r\approx\sum_g w_g f(\mathbf r_g).
```

For a closed-shell density,

```math
\rho(\mathbf r_g)=\sum_{\mu\nu}D_{\mu\nu}\chi_\mu(\mathbf r_g)\chi_\nu(\mathbf r_g).
```

The exchange-correlation energy is approximated by

```math
E_{xc}\approx\sum_g w_g\rho(\mathbf r_g)\varepsilon_{xc}(\rho,\nabla\rho,\ldots).
```

For a local functional, the XC matrix has the schematic form

```math
V^{xc}_{\mu\nu}\approx\sum_g w_gv_{xc}(\mathbf r_g)\chi_\mu(\mathbf r_g)\chi_\nu(\mathbf r_g),
```

with additional gradient-dependent terms for GGA functionals.

## Validation examples

### H2/STO-3G RHF

For H2 near \(R=1.4\) bohr, the RHF energy is approximately

```text
E_RHF approx -1.1167 Ha
```

### H2O/6-31G CISD

The CISD energy has been checked against PySCF and agrees to approximately \(10^{-9}\) Ha in tested cases.

### H2O/6-311G** RHF geometry optimization

The current RHF analytic-gradient BFGS path gives the same optimized geometry as the older finite-difference-gradient workflow while reducing the runtime from tens of seconds to roughly a few seconds on the tested machine.

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

- Add CMake and small regression tests.
- Move single-point RHF, optimization, MP2, CI, and DFT workflows from `main.cpp` into separate driver modules.
- Move finite-difference derivative checks into explicit validation/test executables.
- Centralize basis construction through `BasisContext` everywhere.

Near-term method tasks:

- Validate RKS/LDA/PBE against PySCF.
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
