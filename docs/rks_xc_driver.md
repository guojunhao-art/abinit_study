# Total-XC RKS driver

This stage introduces `rks_xc.hpp/.cpp`, a new restricted Kohn-Sham driver that works with the `XCFunctional` + `XCEvaluator` + `dft_xc_matrix` path.

It is meant to be the new DFT path for B3LYP/PBE0-style hybrid work.  The old `run_rks_lda_exchange_only_sp()` driver is kept for compatibility.

## Why a new driver?

The historical RKS driver stores separate `E_x`, `E_c`, `Vx`, and `Vc` fields.  That is convenient for simple LDA/GGA examples, but it becomes awkward for modern functionals where the useful SCF interface is simply

```text
functional -> total Exc and total Vxc
```

The new driver therefore uses total semilocal `Exc/Vxc` from

```cpp
build_xc_matrix_with_evaluator_sp(...)
```

and separately adds exact exchange when the functional is hybrid.

## Spin-summed density convention

The code uses the closed-shell spin-summed density matrix

```math
D_{\mu\nu}=2\sum_i^{occ}C_{\mu i}C_{\nu i}.
```

With this convention,

```math
J_{\mu\nu}=\sum_{\lambda\sigma}D_{\lambda\sigma}(\mu\nu|\lambda\sigma),
```

and the RHF exchange matrix contribution is

```math
-\frac{1}{2}K.
```

Therefore a full-range hybrid with exact-exchange fraction `a_x` uses

```math
F = H^{core} + J[D] + V_{xc}[D] - \frac{1}{2}a_xK[D].
```

The corresponding exact-exchange energy contribution is

```math
E_x^{HF,hyb}=-\frac{1}{4}a_x\operatorname{Tr}[DK].
```

This is the most important convention in this stage.  If `a_x = 1`, the hybrid Fock exchange term must reduce to the RHF term `-1/2 K`.

## Energy expression

The total-XC RKS driver evaluates

```math
E_{elec}=\operatorname{Tr}[DH^{core}]+\frac{1}{2}\operatorname{Tr}[DJ]+E_{xc}^{semi}+E_x^{HF,hyb}.
```

and

```math
E_{tot}=E_{elec}+E_{nuc}.
```

For a pure semilocal functional, `E_x^{HF,hyb}=0`.

## Current scope

Supported now:

- closed-shell restricted systems;
- pure LDA/GGA semilocal functionals supported by `XCEvaluator`;
- full-range hybrid GGA functionals whose semilocal Libxc part is evaluable by `XCEvaluator`.

Rejected for now:

- unrestricted spin-polarized systems;
- range-separated hybrids;
- meta-GGA functionals such as M06-2X, because tau/vtau terms are not implemented yet.

## Entry point

```cpp
miniqc::rks::RKSXCResult result = miniqc::rks::run_rks_xc(
    basis, S, Hcore, nelec, Enuc, grid, functional, options);
```

The result stores total-energy-oriented fields:

```cpp
E_total
E_elec
E_one
E_coulomb
E_xc
E_exact_exchange
E_nuc
Ne_grid
```

This avoids forcing new hybrid functionals into the old `E_x/E_c` bookkeeping.
