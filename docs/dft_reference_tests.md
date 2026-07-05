# DFT reference tests

This stage adds two levels of DFT testing.

## Default smoke tests

The default CTest suite contains command-line, config-driven RKS tests for:

- `pbe`
- `b3lyp`
- `pbe0`

The corresponding inputs are:

```text
tests/h2_pbe_rks_sto3g.in
tests/h2_b3lyp_rks_sto3g.in
tests/h2_pbe0_rks_sto3g.in
```

These tests only verify that the user-facing executable reaches the final

```text
=== RKS/DFT result ===
```

block.  They intentionally use a small grid and a small maximum number of DFT iterations, because their purpose is to keep the config path from breaking.

## Optional PySCF numerical reference test

The optional test script is:

```text
tests/compare_dft_with_pyscf.py
```

It runs miniqc and PySCF for H2/STO-3G at 1.4 bohr and compares total RKS energies for selected functionals.

Enable it with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DMINIQC_ENABLE_PYSCF_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure -R pyscf
```

This requires PySCF in the Python environment used by CMake:

```bash
python -c "import pyscf; print(pyscf.__version__)"
```

## Cartesian AO convention

miniqc currently constructs Libint basis sets with Cartesian shells:

```cpp
basis.set_pure(false);
```

The PySCF reference script therefore sets:

```python
mol.cart = True
```

This is not important for H2/STO-3G, because the basis only contains s functions.  It becomes essential once the reference set includes p/d functions, such as water/STO-3G or water/6-31G**.

## Tight H2/STO-3G reference settings

For the optional reference test, CTest uses a dense miniqc atom-centered grid:

```text
--n-radial 200
--angular-grid 302
--tolerance 1.0e-6
```

The default script values match these settings.  In local tests with this grid, H2/STO-3G reached approximately `1e-10 Ha` agreement against PySCF for:

```text
pbe
b3lyp
pbe0
```

The CTest tolerance is kept at `1.0e-6 Ha` rather than `1.0e-10 Ha` to avoid overfitting to a specific Libxc/PySCF/BLAS build while still catching real implementation errors.

The reference test should catch:

- missing exact exchange in a hybrid functional;
- a factor-of-two error in `K`;
- wrong functional dispatch;
- broken config-driven RKS path;
- degraded grid construction.

## Manual run

A manual comparison can be run as:

```bash
python tests/compare_dft_with_pyscf.py \
  --miniqc-exe ./build/RHF \
  --workdir ./build/pyscf_reference_work \
  --functionals pbe b3lyp pbe0 \
  --n-radial 200 \
  --angular-grid 302 \
  --tolerance 1.0e-6
```

The output has the form:

```text
miniqc grid: n_radial = 200, angular_grid = 302, r_max = 10.0 bohr
absolute tolerance = 1.000e-06 Ha
pbe       miniqc = ...  pyscf = ...  diff = ...
b3lyp     miniqc = ...  pyscf = ...  diff = ...
pbe0      miniqc = ...  pyscf = ...  diff = ...
PySCF DFT reference comparison passed
```

## Next validation target

The next validation step should add small molecules with p functions, such as water/STO-3G or water/6-31G, because those cases test AO gradient ordering and the Cartesian/spherical basis convention more strongly than H2/STO-3G.
