# chemtrain-deploy long-range benchmark

This project measures how well [chemtrain-deploy](https://chemtrain.readthedocs.io/en/latest/)
runs a JAX machine-learning potential through LAMMPS on multiple GPUs. It uses the **MACE-MH-1**
foundation model and two periodic systems with very different shapes and sizes:

- a DNA duplex in water with salt (DDD), and
- a 400-lipid POPC membrane in water (POPC_Bilayer).

The full chemtrain and chemtrain-deploy documentation is available at
**https://chemtrain.readthedocs.io/en/latest/**, including the LAMMPS interface, model export
format, and foundation-model adapters.

![DDD duplex (left) and POPC bilayer (right) in their periodic simulation cells](benchmark/preview.jpg)
*Left: the DDD DNA duplex with TIP3P water and Na⁺/Cl⁻ ions. Right: the 400-lipid POPC bilayer
with TIP3 water. The input structure for each system is `structure/equilibrated.pdb`.*

## Example Tests

Both tests use the same MACE-MH-1 model and the `omol` model head. The model uses a 6.0 Å cutoff
and two interaction layers. It is exported once as a CUDA chemtrain-deploy bundle, with
OpenEquivariance custom calls and the communication-enabled (`comm on`) graph variant.

LAMMPS runs the model with `pair_style chemtrain` and Kokkos (`-k on -sf kk`). Each MPI rank uses
one GPU, and the GPUs exchange data directly when GPU-aware MPI is available.

The coordinates do not move during the test: the input performs `run 0` and does not integrate
the system. The benchmark repeats the force and energy calculation and measures its throughput.
This measures the performance of LAMMPS, the connector, and the model. 

| | **DDD** | **POPC_Bilayer** |
|---|---|---|
| System | Drew–Dickerson B-DNA dodecamer d(CGCGAATTCGCG)₂, TIP3P water, 150 mM NaCl | 400-lipid CHARMM36 bilayer (200 lipids per leaflet), TIP3 water, no ions |
| Atoms | 18,914 | 107,600 |
| Box | 58.652³ Å cube | 158.2 × 79.1 × 80.8 Å slab |
| Structure | `DDD/structure/equilibrated.pdb` | `POPC_Bilayer/structure/equilibrated.pdb` |
| Built with | OpenMM, amber14/DNA.bsc1 + tip3p (`structure/config.json`) | OpenMM, CHARMM36 + flexible TIP3 (`structure/config.json`, `topol_big.top`, `toppar/`); source: Zenodo [10.5281/zenodo.1198158](https://zenodo.org/records/1198158) (NMRlipids/CHARMM-GUI), replicated 2×1×1 |
| GPU counts tested | 1, 2, 4, 7 | 5, 6, 7  |

The GPU IDs, GPU counts, and `--memory-fraction` values listed here are the values used for this
test machine. See [Site-specific settings](#site-specific-settings) for how to choose them on
another machine.

## Directory layout

```
benchmark/
  run_benchmark.py             runs preparation, model export, and LAMMPS on N GPUs
  prepare_lammps_data.py       converts a periodic PDB file to a LAMMPS data file (ASE)
  export_mace_mh1_omol.py      exports MACE-MH-1 with the omol head
  simulation.lmp               fixed-coordinate throughput benchmark input
  kokkos_neighbor_smoke.lmp    tests Kokkos periodic neighbor-list construction by itself
                                and helps diagnose the CUDA-IPC problem described below
  plot_scaling.py              reads the summary files and box sizes to make the scaling plot
  scaling_speed.png            measured and reference scaling for both systems
  models/
    mace_mh1_omol.ptb          exported model, shared by both systems
    mace_mh1_omol.ptb.json     export metadata: heads, cutoff, layers, and graph variants
  DDD/
    structure/
      equilibrated.pdb          periodic, wrapped input structure with a CRYST1 record
      config.json                OpenMM protocol used to make the structure
    1BNA_DNA.lmpdat             prepared LAMMPS data file; can be regenerated from the PDB
    summary.json / .csv         measured throughput for every GPU count
    *.screen                    raw LAMMPS and mpirun output for each run (gitignored)
  POPC_Bilayer/
    structure/
      equilibrated.pdb          400-lipid bilayer after equilibration
                                  (see its README for the full build procedure)
      config.json                 equilibration protocol
      topol_big.top, toppar/      CHARMM36 topology used to build the structure
                                  (kept for provenance; the benchmark only needs the PDB)
    POPC_bilayer.lmpdat
    summary.json / .csv
    *.screen                    includes the 4-GPU out-of-memory log
build/
  lammps-host/, lammps-kokkos-cuda/, connector/    local LAMMPS and chemtrain-deploy builds
```

The `structure/` directories contain the actual benchmark inputs and are tracked in the
repository. The model files, LAMMPS data files, and raw screen files can be regenerated from
those inputs and are gitignored. The summary files and the small `.ptb.json` export metadata are
kept as the permanent record of the runs.

## Prerequisites

Create a separate environment for the benchmark:

```bash
conda create -n chemtrain-deploy-benchmark python=3.12 -y
conda activate chemtrain-deploy-benchmark
python -m pip install -r requirements.txt
```

The requirements file pins `chemtrain[cuda12]` to a `tummfm/chemtrain` `main` commit. The
`[cuda12]` extra installs JAX CUDA wheels. Update the pin as the project changes; use a tagged release when one becomes available.

The Python dependencies are only part of the setup. The connector and PJRT runtime must be built
separately; follow the
[chemtrain-deploy installation guide](https://chemtrain.readthedocs.io/en/latest/chemtrain-deploy/installation.html).

Next, build a CUDA- and Kokkos-enabled LAMMPS with MPI support and
`PKG_CHEMTRAIN-DEPLOY=ON`. Use the `chemtrain-deploy` branch of
[`github.com/tummfm/lammps`](https://github.com/tummfm/lammps/tree/chemtrain-deploy) (the build
used for these results was commit `553ca5e`). Build it against a CUDA-aware OpenMPI, then point
the benchmark to the executable and connector library:

```bash
export LAMMPS_EXECUTABLE=/path/to/lammps/build/lammps-kokkos-cuda/lmp
export CHEMTRAIN_DEPLOY_LIB=/path/to/lammps/build/connector
```

## Running the benchmark

Run the commands from the `benchmark` directory.

For DDD, prepare the LAMMPS data file and export the model:

```bash
cd benchmark

# Omit --checkpoint to download MACE-MH-1 automatically.
python run_benchmark.py --pdb DDD/structure/equilibrated.pdb \
  --checkpoint ~/.cache/mace/macemh1model --kokkos \
  --cuda-devices <free GPU ids> --gpu-counts 1,2,4
```

For the POPC bilayer, reuse the model exported for DDD and prepare only the new LAMMPS data file:

```bash
python run_benchmark.py --pdb POPC_Bilayer/structure/equilibrated.pdb \
  --data-file POPC_Bilayer/POPC_bilayer.lmpdat --output-directory POPC_Bilayer \
  --skip-export --model models/mace_mh1_omol.ptb \
  --kokkos --cuda-devices <free GPU ids> --gpu-counts 5,6,7 --memory-fraction <see below>
```

`--gpu-counts` can use 1, 2, 4, 5, 6, or 7 GPUs. These are the processor grids defined in
`run_benchmark.py`:

- 1 GPU: `(1,1,1)`
- 2 GPUs: `(2,1,1)`
- 4 GPUs: `(2,2,1)`
- 5, 6, or 7 GPUs: `(P,1,1)`, splitting the longest box direction

There is no 3-GPU grid. Use `--skip-export --model ...` to reuse the exported model and
`--skip-prepare --data-file ...` to reuse a prepared LAMMPS data file. The model is identical for
both systems because it uses the same checkpoint and `omol` head, so export it only once.

The correct `--cuda-devices` and `--memory-fraction` values depend on the GPUs available on your
machine. See [Site-specific settings](#site-specific-settings).

## Results

`benchmark/scaling_speed.png` compares the measured throughput with two reference curves from the
chemtrain-deploy paper:

- **Ideal scaling:** `S(P) = P`. This is the maximum possible scaling with no overhead.
- **Approximate scaling:** Equation 3 from the paper:

  ```
  S(P) = [ (L + 2TR) / (P^(-1/d) L + 2TR) ]^d
  ```

The calculation uses `T = 1` (one cutoff of halo atoms) and `R = 6.0 Å`, and uses each system's
actual `(px, py, pz)` LAMMPS processor grid. This matters because DDD is a cube while the POPC
bilayer is a thin slab.

`T` is 1 even though the model has two interaction layers. These runs use the communication-enabled
(`comm on`) model variant, which exchanges intermediate features between MPI ranks after each
layer. As a result, each rank only needs ghost atoms within one cutoff of its boundaries. A
`comm off` run would use `T = 2`.

After a run, regenerate the plot with:

```bash
python benchmark/plot_scaling.py
```

The script only needs `matplotlib`. It reads the tracked `summary.json` files and the box records
in each `structure/equilibrated.pdb` file.

| System | GPUs | Grid | Atoms/rank | steps/s | atom·steps/s | Eff. vs. ideal | Eff. vs. Eq. 3 |
|---|---:|---|---:|---:|---:|---:|---:|
| DDD | 1 | 1×1×1 | 18,914 | 0.964 | 18,232 | 100% | 100% |
| DDD | 2 | 2×1×1 | 9,457 | 1.563 | 29,560 | 81% | 95% |
| DDD | 4 | 2×2×1 | 4,729 | 2.369 | 44,811 | 61% | 84% |
| DDD | 7 | 7×1×1 | 2,702 | 2.841 | 53,731 | 42% | 85% |
| POPC | 4 | 2×2×1 | 26,900 | — | — | — | **out of memory** |
| POPC | 5 | 5×1×1 | 21,520 | 0.674 | 72,506 | 100% | 100% |
| POPC | 6 | 6×1×1 | 17,933 | 0.791 | 85,124 | 98% | 103% |
| POPC | 7 | 7×1×1 | 15,371 | 0.879 | 94,622 | 93% | 103% |

The DDD results are normalized to the 1-GPU run. The POPC results are normalized to the smallest
working run, which uses 5 GPUs. On 80 GB GPUs, POPC does not fit in the fixed-capacity model graph
with 1–4 GPUs; this is why the 4-GPU run ran out of memory and why 5 GPUs are the POPC baseline.
Larger-memory GPUs may work with fewer GPUs, while smaller-memory GPUs may require more.

DDD reaches 84–85% of the approximate-scaling reference. With 7 GPUs, its 58.65 Å cube is divided
into slabs about 8.4 Å thick, only slightly larger than the 6 Å cutoff. Communication and
neighbor-list overhead therefore matter more than the simple ghost-volume model predicts.

POPC is about 3% above the approximate reference. Its 5-GPU baseline is already limited by
`--memory-fraction 0.95`, close to the GPU memory limit. The model is also partly limited by memory
bandwidth, so its per-atom throughput improves as each GPU receives a smaller partition. The
reference formula assumes a fixed cost per atom and does not include either effect.

See `benchmark/DDD/summary.json` and `benchmark/POPC_Bilayer/summary.json` for wall time, atoms
per rank, and physical GPU IDs. The `.screen` files contain the raw LAMMPS output, including the
out-of-memory traceback.

## Site-specific settings

The reported results were collected on one node with NVIDIA A100-SXM4 80 GB GPUs connected by
NVLink, CUDA 12.6, and CUDA-aware OpenMPI 4.1.5. All runs used one node, with one MPI rank per
GPU.

| Setting | DDD | POPC_Bilayer | How to choose it on another machine |
|---|---|---|---|
| `--cuda-devices` | `0,1,2,3` | Any currently free GPU IDs (`nvidia-smi`). Provide at least as many IDs as the largest GPU count. The IDs used here only show which GPUs were free; GPU 2 was busy. |
| `--gpu-counts` | `1,2,4,7` | Use values defined in `PROCESSOR_GRIDS` in `run_benchmark.py`: `1,2,4,5,6,7`. There is no 3-GPU grid. On 80 GB GPUs, POPC needs at least 5 GPUs; larger GPUs may need fewer and smaller GPUs may need more. |
| `--memory-fraction` | `0.75` | `0.95` | Fraction of each GPU's memory reserved for the XLA BFC memory pool used by the fixed-capacity model graph. POPC needs a larger fraction because each GPU handles more atoms and edges. Start by increasing the value until the graph fits, then reduce it if LAMMPS, Kokkos, or MPI buffers run out of memory. As a rough guide, scale it with atoms per rank and available GPU memory. |

The `MPI_LAUNCHER` CUDA-IPC workaround mentioned in the setup is an OpenMPI issue, not a
hardware-specific setting. It applies to OpenMPI builds that support multiple nodes and is harmless
when the MPI installation does not need it.
