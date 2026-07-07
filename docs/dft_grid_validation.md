# DFT grid validation workflow

This note collects the reference-test workflow used after adding hybrid GGA,
meta-GGA tau/vtau, and high-order Lebedev angular grids.

## Single-grid PySCF comparison

The main optional comparison script is:

```text
tests/compare_dft_with_pyscf.py
```

It supports the aliases:

```text
pbe
blyp
b3lyp
pbe0
m062x
m06-2x
M06-2X
```

It now reports both the total energy error and the integrated grid electron
count:

```text
Ne(grid) = sum_g w_g rho_g
```

A typical M06-2X check is:

```bash
python tests/compare_dft_with_pyscf.py \
  --miniqc-exe ./build/RHF \
  --workdir ./build/pyscf_reference_work \
  --functionals m062x \
  --n-radial 200 \
  --angular-grid 590 \
  --r-max 10.0 \
  --tolerance 5.0e-6
```

For GGA and hybrid GGA functionals, the established tight check remains:

```bash
python tests/compare_dft_with_pyscf.py \
  --miniqc-exe ./build/RHF \
  --workdir ./build/pyscf_reference_work \
  --functionals pbe b3lyp pbe0 \
  --n-radial 200 \
  --angular-grid 302 \
  --r-max 10.0 \
  --tolerance 1.0e-6
```

## Grid convergence scan

The scan script is:

```text
tests/scan_dft_grid_convergence.py
```

It runs miniqc over a grid-parameter list and compares every calculation to a
single PySCF reference energy.  It writes a CSV table with:

```text
functional
n_radial
angular_grid
r_max_bohr
E_miniqc_Ha
E_pyscf_Ha
diff_Ha
abs_diff_Ha
Ne_grid
dNe
abs_dNe
```

A useful M06-2X angular-grid scan is:

```bash
python tests/scan_dft_grid_convergence.py \
  --miniqc-exe ./build/RHF \
  --workdir ./build/grid_scan_work \
  --functionals m062x \
  --n-radial 200 \
  --angular-grid 302 590 770 974 1202 \
  --r-max 10.0 \
  --output ./build/m062x_grid_scan.csv
```

A broader scan is:

```bash
python tests/scan_dft_grid_convergence.py \
  --miniqc-exe ./build/RHF \
  --workdir ./build/grid_scan_work \
  --functionals m062x \
  --n-radial 100 200 300 \
  --angular-grid 302 590 770 974 1202 \
  --r-max 10.0 12.0 15.0 \
  --output ./build/m062x_grid_scan.csv
```

## Recommended starting grids

For H2/STO-3G in the current implementation:

```text
PBE/B3LYP/PBE0: n_radial = 200, angular_grid = 302, r_max = 10 bohr
M06-2X:         n_radial = 200, angular_grid = 590, r_max = 10 bohr
```

For a higher-confidence M06-2X check, use:

```text
n_radial = 300
angular_grid = 974 or 1202
r_max = 10-15 bohr
```

The final tolerance should be chosen from the scan result rather than guessed.
If the energy error and `abs_dNe` both decrease as the grid is refined, the
remaining error is grid dominated.  If the energy error stops decreasing while
`abs_dNe` continues to decrease, inspect the functional convention or AO-grid
matrix contribution.

## Why Ne(grid) is tracked

The integrated electron count is a direct diagnostic of grid quality:

```text
Ne(grid) = sum_g w_g rho_g
```

For H2, the target is 2.  In previous M06-2X tests with `n_radial = 200` and
`angular_grid = 302`, the energy was already close to PySCF but `Ne(grid)` still
showed a few parts in 1e-6 of quadrature error.  Reporting `Ne(grid)` next to the
energy error makes it easier to distinguish grid error from implementation error.

## Next feature: dispersion corrections

Dispersion corrections should be added as a separate energy layer rather than as
part of `XCEvaluator`:

```text
E_total = E_KS + E_disp
```

A good first target is Grimme D3(BJ), because it is pairwise, well documented,
and can be tested independently from the self-consistent XC matrix.  D4 and
VV10-style nonlocal corrections can be considered after the D3 path is stable.
