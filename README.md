# chemtrain-deploy long-range benchmark

Integration and scaling test for [**chemtrain-deploy**](https://chemtrain.readthedocs.io/en/latest/) —
the LAMMPS deployment toolkit for JAX-trained machine-learning potentials — using the
**MACE-MH-1** foundation model on two chemically distinct periodic systems, across 1–7 GPUs.
Full chemtrain/chemtrain-deploy documentation, including the LAMMPS integration, model export
format, and foundation-model adapters, lives at
**https://chemtrain.readthedocs.io/en/latest/** (see its *Chemtrain-Deploy* section). The
underlying parallelization design and cost model referenced below are from the chemtrain-deploy
paper, Fuchs, Chen, Thaler & Zavadlav, *chemtrain-deploy: A parallel and scalable framework for
machine learning potentials in million-atom MD simulations*, [arXiv:2506.04055](https://arxiv.org/abs/2506.04055) /
[J. Chem. Theory Comput. (2025)](https://pubs.acs.org/doi/10.1021/acs.jctc.5c00996).

![DDD duplex (left) and POPC bilayer (right) in their periodic simulation cells](benchmark/preview.jpg)
*Left: the DDD DNA duplex, TIP3P water, Na⁺/Cl⁻ ions. Right: the 400-lipid POPC bilayer, TIP3
water. Both from `structure/equilibrated.pdb` in their respective directories.*

## What's tested

Both benchmarks export the same **MACE-MH-1** multi-head foundation-model checkpoint with its
**`omol`** head selected (6.0 Å cutoff, 2 interaction/message-passing layers), as a CUDA
chemtrain-deploy bundle with OpenEquivariance FFI custom calls and the communication-enabled
(`comm on`) graph variant, then drive it in LAMMPS via `pair_style chemtrain` with Kokkos
(`-k on -sf kk`, device-side GPU-aware MPI communication). Runs hold coordinates fixed
(`run 0`/no time integration) and measure repeated force+energy evaluation throughput — this
isolates LAMMPS/connector/model performance from any question of whether the model is a
validated force field for either system.

| | **DDD** | **POPC_Bilayer** |
|---|---|---|
| System | Drew–Dickerson B-DNA dodecamer d(CGCGAATTCGCG)₂, TIP3P, 150 mM NaCl | 400-lipid CHARMM36 bilayer (200/leaflet), TIP3, no ions |
| Atoms | 18,914 | 107,600 |
| Box | 58.652³ Å (cube) | 158.2 × 79.1 × 80.8 Å (slab) |
| Structure | `DDD/structure/equilibrated.pdb` | `POPC_Bilayer/structure/equilibrated.pdb` |
| Built with | OpenMM, amber14/DNA.bsc1 + tip3p (`structure/config.json`) | OpenMM, CHARMM36 + flexible TIP3 (`structure/config.json`, `topol_big.top`, `toppar/`); source: Zenodo [10.5281/zenodo.1198158](https://zenodo.org/records/1198158) (NMRlipids/CHARMM-GUI), replicated 2×1×1 |
| GPU counts tested | 1, 2, 4, 7 | 5, 6, 7 (4 GPUs → **OOM**, see below) |

## Layout

```
benchmark/
  run_benchmark.py             orchestrates prepare -> export -> LAMMPS runs on N GPUs
  prepare_lammps_data.py       periodic PDB -> LAMMPS data file (ASE)
  export_mace_mh1_omol.py      MACE-MH-1/omol -> CUDA chemtrain-deploy bundle
  simulation.lmp               fixed-coordinate throughput benchmark input
  kokkos_neighbor_smoke.lmp    isolates Kokkos periodic neighbor-list construction from
                                chemtrain-deploy/JAX -- used to diagnose the CUDA-IPC issue below
  scaling_speed.png            measured vs. ideal/approximate (Eq. 3) scaling, both systems
  models/
    mace_mh1_omol.ptb          exported model (identical for both systems, reused as-is)
    mace_mh1_omol.ptb.json     export metadata: heads, r_max, num_interactions, variants
  DDD/
    structure/
      equilibrated.pdb          the actual input: periodic, wrapped, CRYST1 set
      config.json                OpenMM equilibration protocol (force field, water model,
                                  ionic strength, temperature) that produced it
    1BNA_DNA.lmpdat             prepared LAMMPS data (regenerable from structure/equilibrated.pdb)
    summary.json / .csv         measured throughput, all GPU counts
    *.screen                    raw LAMMPS/mpirun output per run (gitignored, kept locally)
  POPC_Bilayer/
    structure/
      equilibrated.pdb          400-lipid bilayer, post-equilibration (see the project's
                                  POPC_Bilayer/README for the full 2x1x1-replication +
                                  flexible-water equilibration pipeline that built this)
      config.json                 equilibration protocol
      topol_big.top, toppar/       the exact CHARMM36 topology used to build it (provenance
                                    only -- the benchmark itself only needs equilibrated.pdb)
    POPC_bilayer.lmpdat
    summary.json / .csv
    *.screen                    includes the 4-GPU OOM failure log
build/
  lammps-host/, lammps-kokkos-cuda/, connector/    local LAMMPS + chemtrain-deploy builds
```

`structure/` is the actual input each benchmark needs and is checked in (small: 1.5 MB / 10 MB
PDBs plus a few KB of topology/config). `models/*.ptb`, `*.lmpdat`, and `*.screen` are
regenerable from it and gitignored; `summary.json/.csv` and the `.ptb.json` metadata are the
durable, small record of each run and stay tracked.

## Prerequisites

```bash
conda activate dev5
python -m pip install -r requirements.txt
```

### Reproducibility outside this machine

Audited directly (git remotes, PyPI, GitHub), not assumed:

| Dependency | Availability |
|---|---|
| LAMMPS + `CHEMTRAIN-DEPLOY` package | public: `github.com/tummfm/lammps`, branch `chemtrain-deploy` |
| `mace-jax` | public: `github.com/tummfm/mace-jax` (`requirements.txt` currently points at a private local fork — swap in this remote) |
| `openequivariance`, `ase`, `mace-torch` | public, plain PyPI, already pinned correctly below |
| MACE-MH-1 checkpoint | no local file needed — `export_mace_mh1_omol.py` downloads it via mace-torch's own foundation-model loader when `--checkpoint` is omitted |
| `chemtrain` (base package) | public: PyPI `chemtrain` / `github.com/tummfm/chemtrain`, but only up to v0.2.0 / `main` |
| **`chemtrain.deploy`** (what this benchmark actually runs) | **not yet public** — only exists on the institutional `gitlab.lrz.de:mfm/science/chemtrain`, branch `chemtrain-deploy-mr`; this exact pinned commit is not on the public GitHub in any branch |

So: everything needed to reproduce this benchmark is publicly available **except**
`chemtrain.deploy` itself, which today requires LRZ/TUM GitLab access. The local `chemtrain`
requirement in `requirements.txt` below is pinned to that private commit for exactly this reason;
if/when `chemtrain-deploy-mr` is pushed to the public `tummfm/chemtrain`, swap in that remote URL
and the rest of this table goes fully public.

Build a CUDA/Kokkos MPI-enabled LAMMPS with `PKG_CHEMTRAIN-DEPLOY=ON` from
`/ds/project/franz/src/lammps/lammps` (or `github.com/tummfm/lammps@chemtrain-deploy` off this
machine) against a CUDA-aware OpenMPI (`/opt/openmpi-4.1.8-cuda` here), and set:

```bash
export LAMMPS_EXECUTABLE=/ds/project/franz/src/chemtrain-deploy-long-range/build/lammps-kokkos-cuda/lmp
export CHEMTRAIN_DEPLOY_LIB=/ds/project/franz/src/chemtrain-deploy-long-range/build/connector
export MPI_LAUNCHER="/opt/openmpi-4.1.8-cuda/bin/mpirun --mca btl_smcuda_use_cuda_ipc 0"
```

The `--mca btl_smcuda_use_cuda_ipc 0` flag is required beyond 2 ranks on this server: OpenMPI's
`smcuda` transport fails to register its shared-memory buffer under the installed CUDA version,
and the next Kokkos kernel then reports `cudaErrorIllegalAddress`. Disabling only CUDA IPC keeps
the CUDA-aware transport active and avoids it (isolated with `kokkos_neighbor_smoke.lmp`, which
reproduces the periodic Kokkos neighbor list without loading chemtrain-deploy/JAX at all).

## Running it

```bash
cd benchmark

# DDD -- builds 1BNA_DNA.lmpdat from the checked-in structure and exports the model fresh
# (omit --checkpoint entirely to download MACE-MH-1 automatically instead)
python run_benchmark.py --pdb DDD/structure/equilibrated.pdb \
  --checkpoint ~/.cache/mace/macemh1model --kokkos \
  --cuda-devices 4,5,6,7 --gpu-counts 1,2,4

# POPC bilayer -- reuse the DDD export (same weights/head), only prepare a new data file
python run_benchmark.py --pdb POPC_Bilayer/structure/equilibrated.pdb \
  --data-file POPC_Bilayer/POPC_bilayer.lmpdat --output-directory POPC_Bilayer \
  --skip-export --model models/mace_mh1_omol.ptb \
  --kokkos --cuda-devices 0,1,3,4,5,6,7 --gpu-counts 5,6,7 --memory-fraction 0.95
```

`--gpu-counts` accepts any of `1, 2, 4, 5, 6, 7` (`PROCESSOR_GRIDS` in `run_benchmark.py`); 1/2/4
split as `(1,1,1)`/`(2,1,1)`/`(2,2,1)`, and 5/6/7 split only the longest box axis as `(P,1,1)`.
The exported model and LAMMPS data file are reused as-is with `--skip-export --model ...` /
`--skip-prepare --data-file ...` — the model is identical across systems (same checkpoint, same
`omol` head), so it only needs exporting once.

## Results

`benchmark/scaling_speed.png` plots measured throughput for both systems against two references
from the chemtrain-deploy paper: **ideal** strong scaling (S(P) = P, the zero-overhead ceiling —
black in the paper's own Figure 2) and the paper's **approximate** strong scaling, Eq. 3,

```
S(P) = [ (L + 2TR) / (P^(-1/d) L + 2TR) ]^d
```

evaluated with this export's T = 2 interaction layers and R = 6.0 Å cutoff, generalized to each
system's actual `(px, py, pz)` LAMMPS grid rather than the paper's cubic `P^(1/d)` shorthand,
since DDD's box is a cube but the bilayer is an anisotropic slab.

| System | GPUs | Grid | Atoms/rank | steps/s | atom·steps/s | Eff. vs. ideal | Eff. vs. Eq. 3 |
|---|---:|---|---:|---:|---:|---:|---:|
| DDD | 1 | 1×1×1 | 18,914 | 0.964 | 18,232 | 100% | 100% |
| DDD | 2 | 2×1×1 | 9,457 | 1.563 | 29,560 | 81% | 105% |
| DDD | 4 | 2×2×1 | 4,729 | 2.369 | 44,811 | 61% | 102% |
| DDD | 7 | 7×1×1 | 2,702 | 2.841 | 53,731 | 42% | 115% |
| POPC | 4 | 2×2×1 | 26,900 | — | — | — | **OOM** |
| POPC | 5 | 5×1×1 | 21,520 | 0.674 | 72,506 | 100% | 100% |
| POPC | 6 | 6×1×1 | 17,933 | 0.791 | 85,124 | 98% | 106% |
| POPC | 7 | 7×1×1 | 15,371 | 0.879 | 94,622 | 93% | 109% |

(DDD normalized to its 1-GPU point, POPC to its smallest working 5-GPU point — 1/2/4 GPUs all
exceed the exported model's fixed-capacity graph on one 80 GB A100 for this system size.)

Takeaways:

- **Measured throughput tracks Eq. 3, not naive linear scaling.** DDD falls to 42% of ideal by
  7 GPUs but stays within 16% of the ghost-atom-aware Eq. 3 prediction throughout — the shortfall
  is the geometry of copying atoms near each domain boundary, not inefficiency.
- **Bigger, thinner-split systems scale better.** The 158 Å-long bilayer, split only along its
  long axis, stays at 93% of true ideal at 7 GPUs — matching the paper's own report that scaling
  is "mostly determined by the effort spent on copied atoms compared to local atoms."
- **Measured throughput exceeding Eq. 3** (both systems, by 6–16% at their higher GPU counts) is
  expected, not a bug: Eq. 3 is a simplified uniform-cost geometric model, not a hard bound — only
  the linear *ideal* line is. The paper reports the same effect for its own Allegro results,
  attributed to XLA making effective use of the extra memory/compute multiple GPUs provide.
- **The 4-GPU OOM is the same cost model predicting its own limit.** Each rank at `2×2×1`
  requested ~86 GB — more than an 80 GB A100 holds — because that decomposition's ghosted domain
  is 37% larger than at 7 GPUs (`(2,2,1)` only halves two axes for a very elongated box, versus
  `(7,1,1)`'s deeper split of the single long axis).

See `benchmark/DDD/summary.json` and `benchmark/POPC_Bilayer/summary.json` for the full
per-run numbers (wall time, per-rank atom counts, physical GPU IDs), and the `.screen` files for
raw LAMMPS output, including the OOM traceback.
