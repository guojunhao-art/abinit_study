# Meta-GGA tau and vtau support

This stage adds the first restricted meta-GGA support to the total-XC RKS path.

The motivating functional is M06-2X, which is a hybrid meta-GGA.  It needs the same ingredients as a GGA plus the kinetic energy density `tau`.

## Density convention

The code uses a spin-summed closed-shell density matrix:

```math
D_{\mu\nu}=2\sum_i^{occ}C_{\mu i}C_{\nu i}.
```

With this convention, the restricted meta-GGA kinetic energy density is implemented as

```math
\tau(\mathbf r)=\frac{1}{2}\sum_{\mu\nu}D_{\mu\nu}\nabla\chi_\mu(\mathbf r)\cdot\nabla\chi_\nu(\mathbf r).
```

The factor `1/2` is important.  Because `D` already contains the closed-shell occupation factor of 2, this expression reduces to

```math
\tau(\mathbf r)=\sum_i^{occ}|\nabla\psi_i(\mathbf r)|^2.
```

## Fock matrix contribution

Libxc returns `vtau`, the derivative of the energy density with respect to `tau`.  Since

```math
\frac{\partial\tau}{\partial D_{\mu\nu}}=\frac{1}{2}\nabla\chi_\mu\cdot\nabla\chi_\nu,
```

the matrix contribution is

```math
V_{\mu\nu}^{\tau}
=\frac{1}{2}\int v_\tau(\mathbf r)\nabla\chi_\mu(\mathbf r)\cdot\nabla\chi_\nu(\mathbf r)d\mathbf r.
```

The grid implementation therefore adds

```cpp
Vxc += weight * 0.5 * vtau * (
    dphidx * dphidx.transpose()
  + dphidy * dphidy.transpose()
  + dphidz * dphidz.transpose()
);
```

## Scope

Supported now:

- restricted LDA/GGA;
- restricted hybrid GGA;
- restricted tau-dependent meta-GGA and hybrid meta-GGA such as M06-2X.

Still not supported:

- unrestricted spin-polarized UKS;
- range-separated hybrids;
- Laplacian-dependent meta-GGAs that need `vlaplacian` matrix terms.

## Tests

The evaluator smoke test now checks the M06-2X Libxc path when `XC_HYB_MGGA_XC_M06_2X` is available.

The command-line smoke test

```text
miniqc_h2_m062x_rks_sto3g
```

runs a short H2/STO-3G M06-2X calculation through the normal config driver.

Numerical reference testing against PySCF should be run manually at first, because meta-GGA grids can be more sensitive than GGA grids.  A typical command is:

```bash
python tests/compare_dft_with_pyscf.py \
  --miniqc-exe ./build/RHF \
  --workdir ./build/pyscf_reference_work \
  --functionals m062x \
  --n-radial 200 \
  --angular-grid 302 \
  --tolerance 1.0e-6
```

If that test fails by a small amount, first scan grid settings before changing the tau/vtau formula.
