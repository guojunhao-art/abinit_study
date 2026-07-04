# Build and regression testing

Stage 5 adds a CMake build and a minimal CTest smoke/regression suite.

## Configure

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

If Libint2 or Libxc are installed in a non-standard location, either activate the conda environment first or pass the paths explicitly:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DLIBINT2_INCLUDE_DIR=$CONDA_PREFIX/include \
  -DLIBINT2_LIBRARY=$CONDA_PREFIX/lib/libint2.so \
  -DLIBXC_INCLUDE_DIR=$CONDA_PREFIX/include \
  -DLIBXC_LIBRARY=$CONDA_PREFIX/lib/libxc.so
```

On some conda-forge installations the Libint library is named `libint2`, while the old manual link command used `-lint2`.  `CMakeLists.txt` searches both names.

## Build

```bash
cmake --build build -j
```

The executable target is named `RHF`, preserving the historical command name.

## Run tests

```bash
ctest --test-dir build --output-on-failure
```

The initial tests are intentionally small:

- `miniqc_template_config`: verifies that the config-file template can be printed.
- `miniqc_h2_rhf_sto3g`: runs H2/STO-3G at `R = 1.4 bohr` and checks that the RHF total energy is near `-1.1167 Ha`.

This is only the first safety net.  After local reference values are fixed, the test suite should be extended to include:

- H2O/STO-3G or 6-31G RHF single point.
- RHF analytic-gradient geometry optimization.
- RHF numerical-gradient geometry optimization.
- One MP2 small-system test.
- One FCI/CID/CISD small-system test.
- One experimental DFT grid smoke test.

## Why this matters

The next planned method work is to expand the DFT module toward common Libxc functionals such as B3LYP and M06-2X, then add DFT analytic gradients.  Those changes will touch the SCF driver, grid code, XC matrix construction, and eventually gradient code, so even a small automated regression suite is useful before starting that refactor.
