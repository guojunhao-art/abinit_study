# XCEvaluator design notes

This stage adds a small `XCEvaluator` layer between `XCFunctional` and the RKS matrix builder.

The purpose is to separate two concerns:

```text
XCFunctional
  describes what the functional is and what ingredients it needs

XCEvaluator
  calls Libxc for rho/sigma/tau blocks and returns exc/vrho/vsigma/vtau
```

The current implementation supports restricted/unpolarized LDA and GGA block evaluation.  It intentionally rejects meta-GGA functionals until tau and vtau matrix contributions are implemented.

## Inputs

For one grid point:

```cpp
struct XCInputPoint {
    double rho = 0.0;
    double sigma = 0.0;
    double laplacian = 0.0;
    double tau = 0.0;
};
```

For a block of grid points:

```cpp
struct XCInputBlock {
    std::vector<double> rho;
    std::vector<double> sigma;
    std::vector<double> laplacian;
    std::vector<double> tau;
};
```

For restricted GGA,

```math
\sigma = \nabla\rho\cdot\nabla\rho.
```

For meta-GGA, a later stage will also provide

```math
\tau(\mathbf r)=\frac{1}{2}\sum_i^{occ}|\nabla\psi_i(\mathbf r)|^2.
```

## Outputs

```cpp
struct XCOutputPoint {
    double exc = 0.0;
    double vrho = 0.0;
    double vsigma = 0.0;
    double vlaplacian = 0.0;
    double vtau = 0.0;
};
```

Here `exc` follows Libxc convention: it is the XC energy per particle.  The grid contribution to the XC energy is therefore

```math
E_{xc} \leftarrow E_{xc} + w_g\rho_g\epsilon_{xc,g}.
```

For LDA, the XC matrix contribution has the form

```math
V_{\mu\nu}^{xc} \leftarrow V_{\mu\nu}^{xc} + w_g v_\rho(g)\chi_\mu(g)\chi_\nu(g).
```

For GGA, the additional sigma derivative contributes schematically as

```math
V_{\mu\nu}^{xc} \leftarrow V_{\mu\nu}^{xc} + w_g\left[v_\rho\chi_\mu\chi_\nu + 2v_\sigma\nabla\rho\cdot\nabla(\chi_\mu\chi_\nu)\right].
```

## Current scope

Supported now:

- restricted LDA: rho only.
- restricted GGA and hybrid-GGA semilocal part: rho + sigma.

Rejected for now:

- unrestricted spin-polarized functionals.
- laplacian-dependent functionals.
- meta-GGA tau-dependent functionals.

This means the evaluator can already be used for PBE and the semilocal Libxc part of B3LYP/PBE0, but M06-2X still needs tau and vtau support before it can be used in the RKS matrix builder.

## Next integration step

The next stage should replace the duplicated Libxc calls inside `dft_lda.cpp` with `evaluate_xc_block()` or with a grid-loop wrapper built on top of this evaluator.

After that, the hybrid RKS Fock can be assembled as

```math
F = H^{core} + J + V_{xc} - a_xK.
```
