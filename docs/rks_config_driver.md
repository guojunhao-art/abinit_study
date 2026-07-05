# Config-driven total-XC RKS path

This stage routes user-facing DFT calculations through the new total-`Exc/Vxc` RKS driver:

```cpp
miniqc::rks::run_rks_xc(...)
```

and removes the old `run_rks_lda_exchange_only_sp()` declarations and implementation from the public DFT helper layer.

## What changed

The old path was:

```text
main.cpp
  -> parse_dft_functional(...)
  -> dft::RKSLDAOptions
  -> dft::run_rks_lda_exchange_only_sp(...)
  -> build_lda_xc_sp(...)
```

The new path is:

```text
main.cpp
  -> miniqc::xc::make_xc_functional(cfg.dft_functional)
  -> miniqc::rks::RKSXCOptions
  -> miniqc::rks::run_rks_xc(...)
  -> dft::build_xc_matrix_with_evaluator_sp(...)
  -> miniqc::xc::evaluate_xc_point(...)
```

This makes the config driver use the same total-XC/hybrid-capable implementation tested in `tests/test_rks_xc_driver.cpp`.

## Why keep `dft_lda.hpp/.cpp`?

The file name is now historical.  It still contains useful DFT grid helper code:

- `GridPoint` and `UniformGridOptions`.
- AO value evaluation on a grid point.
- AO gradient evaluation.
- density gradient and sigma construction.
- legacy semilocal XC matrix helpers used for regression comparisons.

The old SCF driver has been removed from this layer.  New DFT SCF work should use `rks_xc.hpp/.cpp`.

## User-facing input

A DFT single-point input now uses:

```ini
[calculation]
job = single_point
dft = true

[dft]
functional = pbe
```

Hybrid GGA descriptors such as `b3lyp` and `pbe0` are now routed through the new RKS driver as well.  They still need numerical validation against a reference package before being treated as production-ready.

M06-2X is recognized by `XCFunctional`, but it is still rejected by the current RKS path because meta-GGA tau/vtau support is not implemented yet.

## New regression test

`tests/h2_pbe_rks_sto3g.in` runs the executable through the user-facing config path with:

```ini
[dft]
functional = pbe
```

The CTest case `miniqc_h2_pbe_rks_sto3g` verifies that the command-line driver reaches the final RKS/DFT result block.
