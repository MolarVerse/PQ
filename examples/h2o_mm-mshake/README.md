# M-SHAKE: Example with rigid TIP3P water

## Description

This example simulates 2000 rigid TIP3P water molecules (NVT, 298.15 K, 50 ps)
with the M-SHAKE/M-RATTLE algorithm for the intramolecular constraints. It can
be used to check that M-SHAKE gives the right structure and dynamics of
liquid water.

## Input files

- [`input.rst`](input.rst): Start structure and velocities.
- [`moldescriptor.dat`](moldescriptor.dat): Molecule descriptor with the TIP3P charges.
- [`guff.dat`](guff.dat): Intermolecular parameters.
- [`mshake.top`](mshake.top): Rigid reference geometry of one water molecule.
- [`shake.top`](shake.top): Empty topology file (needed by M-SHAKE).
- [`run-01.in`](run-01.in): PQ input file.
- [`O-O.rdf.in`](O-O.rdf.in): PQAnalysis input for the O-O radial distribution function.
- [`O.msd.in`](O.msd.in): PQAnalysis input for the O mean square displacement.

## Running

```bash
PQ run-01.in
rdf O-O.rdf.in
msd O.msd.in
```

`rdf` and `msd` are part of [PQAnalysis](https://github.com/MolarVerse/PQAnalysis).

## Expected results

| Quantity                      | Value              |
|-------------------------------|--------------------|
| mean temperature              | 297 K              |
| first g_OO maximum            | 2.71 at r = 2.77 Å |
| first g_OO minimum            | 0.92 at r = 3.69 Å |
| diffusion coefficient D_total | 5.8e-5 cm²/s       |

SHAKE runs of TIP3P water give the same values within noise. A first g_OO
maximum near 4.5 or D near 1e-6 cm²/s points to the M-RATTLE sign bug from
#916 (fixed in #937).
