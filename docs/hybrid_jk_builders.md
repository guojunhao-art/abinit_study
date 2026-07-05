# Explicit J/K builders for hybrid DFT

This stage adds an explicit exact-exchange builder `build_k_direct()` next to the existing Coulomb builder `build_j_direct()`.

The goal is to prepare the code for hybrid generalized Kohn-Sham calculations.

## Matrix definitions

For the spin-summed closed-shell density matrix used in this code,

```math
D_{\lambda\sigma}=2\sum_i^{occ}C_{\lambda i}C_{\sigma i}.
```

The Coulomb matrix is

```math
J_{\mu\nu}=\sum_{\lambda\sigma}D_{\lambda\sigma}(\mu\nu|\lambda\sigma).
```

The exact-exchange matrix is

```math
K_{\mu\nu}=\sum_{\lambda\sigma}D_{\lambda\sigma}(\mu\lambda|\nu\sigma).
```

The RHF two-electron contribution remains

```math
G^{RHF}=J-\frac{1}{2}K.
```

Because `D` is spin-summed, a full-range hybrid DFT Fock contribution uses

```math
G^{hybrid}=J-\frac{1}{2}a_xK,
```

so the generalized Kohn-Sham Fock matrix becomes

```math
F=H^{core}+J+V_{xc}-\frac{1}{2}a_xK.
```

The exact-exchange energy contribution is

```math
E_x^{HF,hyb}=-\frac{1}{4}a_x\operatorname{Tr}[DK].
```

This convention is important: when `a_x = 1`, the hybrid Fock exchange term must reduce to the RHF exchange term `-1/2 K`.

## Code-level interface

The interface in `two_body_fock.hpp` is:

```cpp
Eigen::MatrixXd build_j_direct(const libint2::BasisSet& basis,
                               const Eigen::MatrixXd& D);

Eigen::MatrixXd build_k_direct(const libint2::BasisSet& basis,
                               const Eigen::MatrixXd& D);

Eigen::MatrixXd build_rhf_g_from_jk(const Eigen::MatrixXd& J,
                                    const Eigen::MatrixXd& K);

Eigen::MatrixXd build_hybrid_g_from_jk(const Eigen::MatrixXd& J,
                                       const Eigen::MatrixXd& K,
                                       double exact_exchange_fraction);

double hybrid_exact_exchange_energy(const Eigen::MatrixXd& D,
                                    const Eigen::MatrixXd& K,
                                    double exact_exchange_fraction);
```

The old `build_g_direct()` function is kept.  It still computes the RHF `J - 1/2 K` contribution in one pass, so existing RHF code does not slow down.

## Why keep `build_g_direct()`?

A naive replacement

```cpp
G = build_j_direct(basis, D) - 0.5 * build_k_direct(basis, D);
```

is useful for testing, but it evaluates two separate four-center integral contractions.  The old RHF path computes the Coulomb-like and exchange-like contributions together inside one shell loop.  Keeping it avoids a performance regression in RHF while still exposing separate `J` and `K` for hybrid DFT.

## Test

`tests/test_jk_builders.cpp` verifies the identity

```math
build\_g\_direct(D) = build\_j\_direct(D) - \frac{1}{2}build\_k\_direct(D)
```

for a small H2/STO-3G density matrix.  It also verifies that `build_hybrid_g_from_jk(J, K, 1.0)` reproduces the RHF two-electron contribution, which guards against accidentally using the wrong spin-density convention.
