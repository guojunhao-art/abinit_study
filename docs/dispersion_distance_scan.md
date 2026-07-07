# Dispersion distance scans

Stage 20 adds a small command-line tool for checking how the additive dispersion
correction behaves as two simple fragments are separated.

The tool is:

```text
tools/scan_dispersion_distance.py
```

It generates a sequence of dimer geometries, runs miniqc at each separation, and
writes a CSV file containing:

```text
R
E_KS_total
E_dispersion
E_total = E_KS_total + E_dispersion
Ne(grid)
```

## Example: methane dimer with PBE-D2

```bash
python tools/scan_dispersion_distance.py \
  --miniqc-exe ./build/RHF \
  --workdir ./build/disp_scan_work \
  --output ./build/methane_pbe_d2_scan.csv \
  --model methane \
  --functional pbe \
  --dispersion d2 \
  --distances 3.5 4.0 4.5 5.0 6.0 7.0 8.0 10.0 \
  --basis sto-3g \
  --n-radial 80 \
  --angular-grid 50 \
  --r-max 12.0
```

The stderr progress lines look like:

```text
R= 3.500 A  E_KS= ...  E_disp= ...  E_total= ...
R= 4.000 A  E_KS= ...  E_disp= ...  E_total= ...
```

The CSV contains the same values in machine-readable form.

## Built-in models

Supported built-in dimer models:

```text
methane
water
h2
```

The `distance_angstrom` value is the center-to-center separation between the two
fragments.  These geometries are intentionally simple and are meant for checking
qualitative behavior rather than for benchmark-quality noncovalent interaction
energies.

## Why this is useful

For D2, the long-range pair energy behaves approximately as:

```text
E_disp ~ -C6/R^6
```

At short range the damping function suppresses this term to avoid double counting
with the semilocal exchange-correlation functional and to prevent the raw
`R^-6` expression from becoming unphysically large.

A distance scan lets us check the expected behavior directly:

- `E_dispersion` should be negative for attractive dispersion.
- Its magnitude should decrease rapidly as the fragments are separated.
- The corrected total energy should equal `E_KS_total + E_dispersion`.

## Comparing no-dispersion and D2 curves

To isolate the D2 contribution, run the same scan twice:

```bash
python tools/scan_dispersion_distance.py \
  --miniqc-exe ./build/RHF \
  --workdir ./build/disp_scan_none \
  --output ./build/methane_pbe_none_scan.csv \
  --model methane \
  --functional pbe \
  --dispersion none

python tools/scan_dispersion_distance.py \
  --miniqc-exe ./build/RHF \
  --workdir ./build/disp_scan_d2 \
  --output ./build/methane_pbe_d2_scan.csv \
  --model methane \
  --functional pbe \
  --dispersion d2
```

The difference between the corrected D2 curve and the no-dispersion KS curve is
the additive dispersion energy, up to any small SCF/grid differences caused by
using separate runs.

## Current limitations

- The script is a diagnostic utility, not a benchmark generator.
- The built-in dimers are fixed-orientation toy geometries.
- No counterpoise correction is applied.
- No monomer subtraction is done automatically yet.

A later version can extend this to interaction energies:

```text
Delta E_int(R) = E_dimer(R) - E_monomer(A) - E_monomer(B)
```

and optionally apply counterpoise correction for basis-set superposition error.
