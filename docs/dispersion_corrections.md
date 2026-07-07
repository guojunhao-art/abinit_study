# Dispersion corrections

## Why a correction is needed

Local and semilocal density functionals use information such as the density,
density gradient, and sometimes the kinetic-energy density at each grid point.
This makes them efficient, but it also means they do not naturally recover the
asymptotic London dispersion attraction between two distant neutral fragments.
At long range, the leading two-body dispersion term behaves like

```text
E_disp ~ - C6 / R^6
```

Many practical DFT calculations therefore add an empirical or semiempirical
post-SCF correction:

```text
E_total = E_KS + E_disp
```

In the current miniqc implementation, the dispersion layer is additive.  It does
not enter the Fock/Kohn-Sham matrix and therefore does not change the SCF density
for a single-point calculation.

## DFT-D2 model implemented here

Stage 19 implements the Grimme D2 two-body correction as the first pedagogical
version.  For atoms `i` and `j`,

```text
E_D2 = - s6 sum_{i<j} C6_ij / R_ij^6 * f_damp(R_ij)
```

with

```text
C6_ij = sqrt(C6_i C6_j)
R0_ij = R0_i + R0_j
f_damp(R) = 1 / (1 + exp[-d * (R/R0_ij - 1)])
```

The tabulated D2 atomic parameters are:

```text
C6_i: J nm^6 mol^-1
R0_i: Angstrom
```

The implementation converts `C6_i` to atomic units (`Hartree bohr^6`) and
converts `R0_i` to bohr before evaluating the energy.

The default damping steepness is:

```text
d = 20
```

Supported elements in this first implementation:

```text
H, C, N, O, F, P, S, Cl, Br, I
```

Supported built-in functional scale factors:

```text
PBE   s6 = 0.75
BLYP  s6 = 1.20
B3LYP s6 = 1.05
PBE0  s6 = 0.60
```

For other functionals, set `dispersion.s6` explicitly.

## Input syntax

Dispersion is off by default.

```ini
[dispersion]
method = none
```

Enable PBE-D2 with the built-in PBE scale factor:

```ini
[dft]
functional = pbe

[dispersion]
method = d2
```

Set the scale factor manually:

```ini
[dft]
functional = m06-2x

[dispersion]
method = d2
s6 = 0.50
damping_d = 20.0
```

## Output

When dispersion is disabled, the RKS block prints the usual total energy:

```text
E_total = E_KS
```

When dispersion is enabled, the output includes the additive correction:

```text
dispersion       = d2
dispersion_s6    = ...
E_dispersion     = ... Ha
E_KS_total       = ... Ha
E_total          = E_KS_total + E_dispersion
```

## Why D2 first, not D3/D4 first?

D2 is simple enough to implement and test independently from the SCF machinery.
It uses fixed atomic `C6` and `R0` values, so the correction is just a pairwise
geometry-dependent energy.  That makes it a useful first step for this teaching
code.

D3 improves on D2 by making dispersion coefficients environment dependent through
coordination numbers and by adding an optional `C8/R^8` contribution.  D3(BJ)
uses Becke-Johnson damping, which behaves differently from the original D2/D3
zero damping at short range.

D4 goes further by making polarizabilities and dispersion coefficients charge
dependent.  That is more transferable, but it also requires additional machinery
for atomic charges or charge-equilibration-like models.

VV10/rVV10 is a different route: it is a nonlocal correlation functional rather
than a simple pairwise post-SCF correction.  It belongs closer to the numerical
XC grid layer and is a larger project.

## Current limitations

- Energy only; no analytic dispersion gradient yet.
- No periodic boundary conditions or pair cutoffs yet.
- D2 parameters only for a small element set useful for organic test molecules.
- D2 is an older correction model; it is included first for clarity and testing.
  A D3(BJ) implementation should be the next production-quality target.
