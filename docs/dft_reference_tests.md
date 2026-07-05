# DFT reference tests

This stage adds two levels of DFT testing.

## Default smoke tests

The default CTest suite now contains command-line, config-driven RKS tests for:

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

## Why the tolerance is loose

The CTest registration currently uses an absolute tolerance of `5.0e-2 Ha`:

```text
--tolerance 5.0e-2
```

This is intentionally loose.  The current miniqc DFT implementation still uses an educational atom-centered grid and has not been tuned against PySCF's production-quality grids.  At this stage the reference test is meant to catch large mistakes:

- missing exact exchange in a hybrid functional;
- a factor-of-two error in `K`;
- wrong functional dispatch;
- broken config-driven RKS path.

After the grid implementation is improved, this tolerance should be tightened.

## Manual run

A manual comparison can be run as:

```bash
python tests/compare_dft_with_pyscf.py \
  --miniqc-exe ./build/RHF \
  --workdir ./build/pyscf_reference_work \
  --functionals pbe b3lyp pbe0 \
  --tolerance 5.0e-2
```

The output has the form:

```text
pbe       miniqc = ...  pyscf = ...  diff = ...
b3lyp     miniqc = ...  pyscf = ...  diff = ...
pbe0      miniqc = ...  pyscf = ...  diff = ...
PySCF DFT reference comparison passed
```

## Next validation target

Once H2/STO-3G is stable, the next validation step should add small molecules with p functions, such as water/STO-3G or water/6-31G, because those cases test AO gradient ordering and the Cartesian/spherical basis convention more strongly.
