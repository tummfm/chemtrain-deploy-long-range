# SPDX-License-Identifier: Apache-2.0
"""Plot measured strong scaling for the DDD and POPC_Bilayer benchmarks against
the two references from the chemtrain-deploy paper (arXiv:2506.04055):

  * "Ideal" strong scaling  -- S(P) = P, the zero-overhead ceiling, drawn black
    in the paper's Figure 2.
  * "Approximate" strong scaling, Eq. 3 --

        S(P) = [ (L + 2TR) / (P^(-1/d) L + 2TR) ]^d

    the cost of a semi-local model being proportional to (L + 2TR)^d once copied
    (ghost) atoms within T*R of every subdomain face are included, where T is the
    halo depth in cutoffs. The benchmark runs the *communication-enabled*
    (`comm on`) model variant: intermediate features are exchanged between ranks
    after every message-passing step, so a rank only needs ghost atoms within one
    cutoff of its faces rather than the num_interactions * r_max receptive field.
    Hence T = 1 here, not the 2 message-passing layers of the omol head, with
    R = 6.0 Aa. Generalised to each system's actual (px, py, pz) LAMMPS grid,

        cost(grid) = prod_axis ( L_axis / p_axis + 2TR ),

    rather than the paper's cubic P^(1/d) shorthand -- DDD's cell is a cube but
    the bilayer is an anisotropic slab and both are split along one axis at high
    GPU counts.

Reads benchmark/DDD/summary.json and benchmark/POPC_Bilayer/summary.json (the
tracked record of each run) plus the CRYST1 box from each structure/equilibrated.pdb.
Writes benchmark/scaling_speed.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent

T = 1      # halo depth in cutoffs: the comm-on variant exchanges features every
           # step, so one cutoff of ghosts suffices (not the 2 MP layers of omol)
R = 6.0    # model cutoff in Angstrom
TWO_TR = 2.0 * T * R

SYSTEMS = [
    {
        "key": "DDD",
        "title": "DDD  (Drew–Dickerson B-DNA, 18,914 atoms)",
        "color": "#1f77b4",
    },
    {
        "key": "POPC_Bilayer",
        "title": "POPC bilayer  (400 lipids, 107,600 atoms)",
        "color": "#d62728",
    },
]


def read_box(pdb_path: Path) -> tuple[float, float, float]:
    """Return (Lx, Ly, Lz) in Angstrom from the CRYST1 record."""
    with pdb_path.open() as handle:
        for line in handle:
            if line.startswith("CRYST1"):
                return float(line[6:15]), float(line[15:24]), float(line[24:33])
    raise ValueError(f"no CRYST1 record in {pdb_path}")


def ghost_cost(box: tuple[float, float, float], grid: list[int]) -> float:
    """prod_axis (L_axis / p_axis + 2TR) -- work per rank incl. copied atoms."""
    cost = 1.0
    for length, procs in zip(box, grid):
        cost *= length / procs + TWO_TR
    return cost


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    for ax, system in zip(axes, SYSTEMS):
        summary = json.loads(
            (SCRIPT_DIR / system["key"] / "summary.json").read_text()
        )
        box = read_box(SCRIPT_DIR / system["key"] / "structure" / "equilibrated.pdb")

        runs = sorted(summary["results"], key=lambda r: r["gpu_count"])
        ref = runs[0]
        p_ref = ref["gpu_count"]
        tput_ref = ref["atom_steps_per_second"]
        cost_ref = ghost_cost(box, ref["processor_grid"])

        p_meas = [r["gpu_count"] for r in runs]
        s_meas = [r["atom_steps_per_second"] / tput_ref for r in runs]

        # Dense P range for the reference curves.
        p_lo, p_hi = p_ref, max(p_meas)
        p_line = [p_lo + i * (p_hi - p_lo) / 200 for i in range(201)]
        s_ideal = [p / p_ref for p in p_line]

        # Eq. 3 needs an integer processor grid; evaluate at the tested grids.
        grid_by_p = {r["gpu_count"]: r["processor_grid"] for r in runs}
        p_grid = sorted(grid_by_p)
        s_approx = [cost_ref / ghost_cost(box, grid_by_p[p]) for p in p_grid]

        ax.plot(
            p_line, s_ideal, "--", color="0.25", lw=1.6,
            label="ideal  $S(P)=P$",
        )
        ax.plot(
            p_grid, s_approx, ":", color=system["color"], lw=1.8, marker="s",
            ms=5, mfc="white", label="approximate  ($S(P)$ from ghost-atom cost)",
        )
        ax.plot(
            p_meas, s_meas, "-o", color=system["color"], lw=2.0, ms=7,
            label="measured (MACE-MH-1 / omol)",
        )

        for p, s in zip(p_meas, s_meas):
            ax.annotate(
                f"{s / (p / p_ref) * 100:.0f}%",
                (p, s), textcoords="offset points", xytext=(6, -12),
                fontsize=8, color=system["color"],
            )

        ax.set_title(system["title"], fontsize=10)
        ax.set_xlabel("GPUs  $P$")
        ax.set_ylabel(f"speed-up  $S(P)$   (vs. {p_ref} GPU"
                      f"{'s' if p_ref > 1 else ''})")
        ax.set_xticks(p_meas)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "chemtrain-deploy strong scaling  —  fixed-coordinate force+energy "
        "throughput, Kokkos/CUDA, GPU-aware MPI (comm on)",
        fontsize=11,
    )
    fig.text(
        0.5, 0.005,
        "% labels: parallel efficiency $\\varepsilon = S/P$ "
        "relative to the per-panel reference point.",
        ha="center", fontsize=8, color="0.35",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    out = SCRIPT_DIR / "scaling_speed.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
