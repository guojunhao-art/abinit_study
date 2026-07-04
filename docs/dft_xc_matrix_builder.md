# DFT XC matrix builder with XCEvaluator

This stage adds `dft_xc_matrix.hpp/.cpp`, a bridge between the new `XCEvaluator` layer and the existing AO-grid matrix assembly code.

It is a conservative integration step.  The historical `build_lda_xc_sp()` path is still kept, but the new builder is tested against it for LDA_X and PBE.

## Interface

```cpp
LDAExchangeResult
build_xc_matrix_with_evaluator_sp(const libint2::BasisSet& basis,
                                  const Eigen::MatrixXd& D,
                                  const std::vector<GridPoint>& grid,
                                  const miniqc::xc::XCFunctional& functional,
                                  double rho_cutoff = 1.0e-14);
```

Inputs:

- `basis`: AO basis.
- `D`: spin-summed closed-shell density matrix.
- `grid`: real-space grid points and weights.
- `functional`: `XCFunctional` descriptor.
- `rho_cutoff`: skip points with negligible density.

Output:

- `LDAExchangeResult`, using the historical result container.
- `Exc` and `Vxc` are the total semilocal XC energy and matrix.
- `Ex/Vx` are currently filled with the same total semilocal contribution for compatibility.
- `Ec/Vc` are set to zero in this new total-only path.

## LDA matrix term

For LDA, the evaluator returns `exc` and `vrho`.  The energy contribution is

```math
E_{xc} \leftarrow E_{xc} + w_g \rho_g \epsilon_{xc,g}.
```

The matrix contribution is

```math
V_{\mu\nu}^{xc} \leftarrow V_{\mu\nu}^{xc} + w_g v_\rho(g)\chi_\mu(g)\chi_\nu(g).
```

## GGA matrix term

For restricted GGA,

```math
\sigma = \nabla\rho\cdot\nabla\rho.
```

The evaluator returns `exc`, `vrho`, and `vsigma`.  The matrix contribution is

```math
V_{\mu\nu}^{xc} \leftarrow V_{\mu\nu}^{xc} + w_g\left[v_\rho\chi_\mu\chi_\nu + 2v_\sigma\nabla\rho\cdot\nabla(\chi_\mu\chi_\nu)\right].
```

The implemented code constructs

```cpp
gdot_mu = grad_rho_x * dphi_mu_dx
        + grad_rho_y * dphi_mu_dy
        + grad_rho_z * dphi_mu_dz;

grad_term_mn = gdot_m * phi_n + phi_m * gdot_n;
```

Then adds

```cpp
Vxc += w * vrho * (phi * phi.transpose());
Vxc += w * 2.0 * vsigma * grad_term;
```

## Why this stage does not yet replace the RKS driver

The current RKS driver still expects historical `Ex/Ec/Vx/Vc` bookkeeping.  The new builder is total-XC oriented and is better suited to the future `XCFunctional`/hybrid path.  To avoid changing SCF energies and printed fields in the same PR, this stage only adds the new builder and tests it against the old implementation.

The next stage can safely make `build_lda_xc_sp()` call this builder for Libxc LDA/GGA functionals, or introduce a new RKS driver that works directly with total `Exc/Vxc`.

## Test

`tests/test_dft_xc_matrix.cpp` verifies that the new evaluator-based builder reproduces the old Libxc builder for:

- LDA_X.
- PBE.

The test compares total `Exc`, `Ne_grid`, and `Vxc` on a small H2/STO-3G grid.
