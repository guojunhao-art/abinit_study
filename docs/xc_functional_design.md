# XCFunctional design notes

This stage introduces an `XCFunctional` descriptor layer.  It is not yet a full B3LYP/M06-2X implementation.  Its purpose is to make the next DFT steps explicit and avoid adding more hard-coded branches inside the current RKS driver.

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

The current PR adds the first layer only.

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

For a full-range hybrid functional:

```math
F = H^{core} + J[D] + V_{xc}[D] - a_x K[D].
```

The direct two-electron helper layer now exposes explicit `J` and `K` builders:

```cpp
build_j_direct(basis, D);
build_k_direct(basis, D);
```

RHF can still use

```math
G^{RHF} = J - \frac{1}{2}K.
```

Hybrid DFT will use

```math
G^{hybrid} = J - a_xK.
```

The old `build_g_direct()` RHF helper is still kept because it computes `J - 1/2 K` in one shell loop and avoids slowing down existing RHF calculations.

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

The current RKS code still supports only the old matrix builder path:

- Slater exchange.
- LDA_X.
- LDA_X + PZ81 correlation.
- PBE.

B3LYP, PBE0, and M06-2X are now describable by `XCFunctional`, but they still require the next implementation steps:

1. Refactor the RKS driver to assemble hybrid generalized-KS Fock matrices.
2. Add an `XCEvaluator` that handles LDA/GGA/meta-GGA ingredients uniformly.
3. Add tau and vtau terms for meta-GGA functionals.
