# XCFunctional design notes

This stage introduces an `XCFunctional` descriptor layer.  It is not yet a full M06-2X implementation.  Its purpose is to make the next DFT steps explicit and avoid adding more hard-coded branches inside the current RKS driver.

## Layering

The intended DFT stack is:

```text
XCFunctional
  describes the functional: family, Libxc components, exact exchange, required ingredients

XCEvaluator
  calls Libxc on blocks of grid data: rho, sigma, tau, ...

RKS / UKS driver
  builds J, optional K, Vxc, diagonalizes the KS Fock matrix, runs DIIS
```

## Functional families

```cpp
enum class XCFamily {
    LDA,
    GGA,
    MGGA,
    HybridGGA,
    HybridMGGA,
    RangeSeparatedHybridGGA,
    RangeSeparatedHybridMGGA
};
```

The family determines what the SCF code must be able to build.

| Family | Needed density ingredients | Extra Fock term |
|---|---|---|
| LDA | rho | none |
| GGA | rho, sigma | none |
| meta-GGA | rho, sigma, tau | vtau contribution |
| hybrid GGA | rho, sigma | exact exchange K |
| hybrid meta-GGA | rho, sigma, tau | exact exchange K + vtau contribution |

## Requirements

Each functional carries an `XCRequirements` object:

```cpp
struct XCRequirements {
    bool needs_rho = true;
    bool needs_sigma = false;
    bool needs_laplacian = false;
    bool needs_tau = false;
};
```

Examples:

- LDA_X: `needs_rho = true`.
- PBE: `needs_rho = true`, `needs_sigma = true`.
- B3LYP: same as GGA plus `exact_exchange_fraction = 0.20`.
- M06-2X: needs sigma and tau plus `exact_exchange_fraction = 0.54`.

## Exact exchange

For pure RKS:

```math
F = H^{core} + J[D] + V_{xc}[D].
```

The code uses a spin-summed closed-shell density matrix,

```math
D_{\mu\nu}=2\sum_i^{occ}C_{\mu i}C_{\nu i}.
```

With this convention, a full-range hybrid functional uses

```math
F = H^{core} + J[D] + V_{xc}[D] - \frac{1}{2}a_x K[D].
```

The helper layer exposes explicit `J` and `K` builders:

```cpp
build_j_direct(basis, D);
build_k_direct(basis, D);
```

RHF uses

```math
G^{RHF} = J - \frac{1}{2}K.
```

Hybrid DFT uses

```math
G^{hybrid} = J - \frac{1}{2}a_xK.
```

The exact-exchange energy contribution is

```math
E_x^{HF,hyb}=-\frac{1}{4}a_x\operatorname{Tr}[DK].
```

## M06-2X and tau

M06-2X is a hybrid meta-GGA.  Besides exact exchange, it needs the kinetic energy density:

```math
\tau(\mathbf r)=\frac{1}{2}\sum_i^{occ}|\nabla\psi_i(\mathbf r)|^2.
```

With a closed-shell AO density matrix, this is implemented from AO gradients as a contraction of the form

```math
\tau(\mathbf r)\sim\frac{1}{2}\sum_{\mu\nu}D_{\mu\nu}\nabla\chi_\mu(\mathbf r)\cdot\nabla\chi_\nu(\mathbf r),
```

with the prefactor kept consistent with the definition of the spin-summed density matrix.

## Current support boundary

The new total-XC RKS driver can handle closed-shell restricted LDA/GGA and full-range hybrid-GGA functionals whose semilocal Libxc part is supported by `XCEvaluator`.

Still missing:

1. connect the new `run_rks_xc()` path to the user-facing config driver;
2. add stronger numerical regression tests against PySCF or another reference;
3. add tau and vtau terms for meta-GGA functionals such as M06-2X;
4. add unrestricted UKS/UHF later.
