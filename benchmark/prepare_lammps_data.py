#!/usr/bin/env python3
"""Convert a periodic PDB structure into the benchmark's LAMMPS data file."""

from __future__ import annotations

import argparse
from pathlib import Path

from ase.io import read, write


SPECORDER = ("H", "C", "N", "O", "Na", "P", "Cl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Periodic PDB file")
    parser.add_argument("--output", type=Path, required=True, help="LAMMPS data file")
    args = parser.parse_args()

    atoms = read(args.input)
    if len(atoms) == 0:
        raise ValueError(f"No atoms found in {args.input}")
    if not all(atoms.pbc):
        raise ValueError(
            f"{args.input} does not declare periodic boundary conditions; "
            "the long-range benchmark requires a periodic cell."
        )
    if atoms.cell.rank != 3:
        raise ValueError(f"{args.input} does not contain a full 3-D periodic cell")

    symbols = set(atoms.get_chemical_symbols())
    unexpected = symbols.difference(SPECORDER)
    if unexpected:
        raise ValueError(
            "The benchmark type map does not cover elements: "
            + ", ".join(sorted(unexpected))
        )

    # The PDB contains valid periodic coordinates outside the primary cell.
    # Wrap them once here; LAMMPS then applies the same periodic cell when it
    # builds the neighbor list.
    atoms.wrap()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(
        args.output,
        atoms,
        format="lammps-data",
        atom_style="atomic",
        specorder=list(SPECORDER),
        masses=True,
        units="metal",
    )
    print(
        f"Prepared {len(atoms)} atoms, cell={atoms.cell.lengths().tolist()} Å, "
        f"pbc={atoms.pbc.tolist()}, output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
