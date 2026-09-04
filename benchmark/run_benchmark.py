#!/usr/bin/env python3
"""Export MACE-MH-1/OMOL and benchmark it in LAMMPS on 1-7 GPUs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_PDB = Path(
    "/ds/project/franz/projects/LongRange/Drew–Dickerson/openmm/runs/"
    "1BNA_DNA.bsc1_tip3p_150mM_300K_rep0/equilibrated.pdb"
)
PROCESSOR_GRIDS = {1: (1, 1, 1), 2: (2, 1, 1), 4: (2, 2, 1),
                   5: (5, 1, 1), 6: (6, 1, 1), 7: (7, 1, 1)}
LOOP_PATTERN = re.compile(
    r"Loop time of\s+([0-9.eE+-]+)\s+on\s+(\d+)\s+procs\s+for\s+(\d+)\s+steps"
)
ATOM_PATTERN = re.compile(r"^\s*(\d+)\s+atoms\s*$", re.MULTILINE)


def run_checked(
    name: str,
    command: list[str],
    *,
    environment: dict[str, str],
    output_directory: Path,
) -> str:
    """Run a command and save its combined screen output."""
    print(f"\n[{name}] {' '.join(shlex.quote(part) for part in command)}", flush=True)
    screen_path = output_directory / f"{name}.screen"
    # MPI/CUDA compiler descendants can retain an inherited pipe descriptor
    # after mpirun exits. Writing directly to the screen file avoids waiting
    # indefinitely for pipe EOF and also avoids buffering large PTX logs.
    with screen_path.open("w") as screen_handle:
        completed = subprocess.run(
            command,
            cwd=SCRIPT_DIRECTORY,
            env=environment,
            text=True,
            stdout=screen_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    screen = screen_path.read_text()
    print(screen, end="", flush=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code {completed.returncode}; "
            f"see {screen_path}"
        )
    return screen


def parse_measurement(screen: str, requested_steps: int) -> dict[str, float | int]:
    """Use the last nonzero LAMMPS loop as the measured section."""
    loops = [
        (float(seconds), int(processes), int(steps))
        for seconds, processes, steps in LOOP_PATTERN.findall(screen)
        if int(steps) > 0
    ]
    if not loops:
        raise RuntimeError("No nonzero LAMMPS loop was found in the screen output")
    seconds, processes, steps = loops[-1]
    if steps != requested_steps:
        raise RuntimeError(
            f"Measured LAMMPS loop has {steps} steps, expected {requested_steps}"
        )
    atom_match = ATOM_PATTERN.search(screen)
    atoms = int(atom_match.group(1)) if atom_match else None
    result: dict[str, float | int] = {
        "elapsed_seconds": seconds,
        "mpi_ranks": processes,
        "steps": steps,
        "steps_per_second": steps / seconds,
    }
    if atoms is not None:
        result["atoms"] = atoms
        result["atom_steps_per_second"] = atoms * steps / seconds
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, default=DEFAULT_PDB)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--lmp", default=os.environ.get("LAMMPS_EXECUTABLE", "lmp"))
    parser.add_argument(
        "--mpi-launcher",
        default=os.environ.get("MPI_LAUNCHER", "mpirun"),
        help="MPI launcher and optional arguments before -np",
    )
    parser.add_argument(
        "--kokkos",
        action="store_true",
        help="Use chemtrain/kk with CUDA-aware device MPI communication.",
    )
    parser.add_argument(
        "--kokkos-binsize",
        type=float,
        default=None,
        help="Optional absolute Kokkos neighbor-bin size in Angstrom.",
    )
    parser.add_argument(
        "--gpu-counts",
        default="1,2,4",
        help="Comma-separated subset of the supported GPU counts: 1,2,4.",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--memory-fraction", type=float, default=0.75)
    parser.add_argument("--atom-capacity", type=float, default=1.05)
    parser.add_argument("--edge-capacity", type=float, default=1.05)
    parser.add_argument(
        "--cuda-devices",
        default=os.environ.get("CUDA_DEVICES", "4,5,6,7"),
        help="Comma-separated physical GPU IDs; the first 1/2/4 are used.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Reuse the existing MACE-MH-1/OMOL bundle.",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Reuse the existing LAMMPS data file.",
    )
    args = parser.parse_args()

    if args.steps <= 0 or args.warmup_steps < 0 or args.repeats <= 0:
        raise ValueError("steps must be positive; warmup and repeats must be nonnegative")
    if not 0.0 < args.memory_fraction <= 1.0:
        raise ValueError("memory-fraction must be in (0, 1]")
    if args.atom_capacity < 1.0 or args.edge_capacity < 1.0:
        raise ValueError("atom-capacity and edge-capacity must be at least 1")
    if args.kokkos_binsize is not None and args.kokkos_binsize <= 0.0:
        raise ValueError("kokkos-binsize must be positive")
    gpu_counts = [int(value) for value in args.gpu_counts.split(",")]
    if not gpu_counts or len(set(gpu_counts)) != len(gpu_counts):
        raise ValueError("gpu-counts must contain distinct values")
    if any(value not in PROCESSOR_GRIDS for value in gpu_counts):
        raise ValueError(
            f"gpu-counts supports only {sorted(PROCESSOR_GRIDS)}")
    devices = [device.strip() for device in args.cuda_devices.split(",") if device.strip()]
    if len(devices) < max(gpu_counts, default=0):
        raise ValueError(
            f"--cuda-devices must list at least {max(gpu_counts)} physical GPU IDs "
            f"for a {max(gpu_counts)}-GPU run")

    output_directory = args.output_directory or SCRIPT_DIRECTORY / "DDD"
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    model = (
        args.model.resolve()
        if args.model is not None
        # The exported model is identical for every system (same weights,
        # same head), so it lives once in models/ rather than per-system.
        else SCRIPT_DIRECTORY / "models" / "mace_mh1_omol.ptb"
    )
    data_file = (
        args.data_file.resolve()
        if args.data_file is not None
        else output_directory / "1BNA_DNA.lmpdat"
    )
    input_file = SCRIPT_DIRECTORY / "simulation.lmp"

    base_environment = os.environ.copy()
    base_environment.update(
        {
            "NVIDIA_TF32_OVERRIDE": "0",
            "JAX_DEFAULT_MATMUL_PRECISION": "float32",
            "CUDA_MODULE_LOADING": os.environ.get("CUDA_MODULE_LOADING", "LAZY"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get(
                "XLA_PYTHON_CLIENT_PREALLOCATE", "false"
            ),
            "JCN_VALIDATE_COMMUNICATION": os.environ.get(
                "JCN_VALIDATE_COMMUNICATION", "1"
            ),
        }
    )
    deploy_lib = os.environ.get("CHEMTRAIN_DEPLOY_LIB")
    if deploy_lib:
        deploy_lib_path = Path(deploy_lib).resolve()
        base_environment["LD_LIBRARY_PATH"] = ":".join(
            [str(deploy_lib_path), base_environment.get("LD_LIBRARY_PATH", "")]
        ).rstrip(":")
        base_environment["JCN_PJRT_PATH"] = str(deploy_lib_path / "pjrt")
        # The connector appends the active backend (for example ``cuda``) to
        # each FFI provider search root.
        base_environment["JCN_FFI_PATH"] = str(deploy_lib_path / "ffi")

    python = sys.executable
    if not args.skip_prepare:
        run_checked(
            "prepare",
            [
                python,
                str(SCRIPT_DIRECTORY / "prepare_lammps_data.py"),
                "--input",
                str(args.pdb.resolve()),
                "--output",
                str(data_file),
            ],
            environment=base_environment,
            output_directory=output_directory,
        )
    elif not data_file.is_file():
        raise FileNotFoundError(f"--skip-prepare requested but missing {data_file}")

    if args.skip_export:
        if not model.is_file():
            raise FileNotFoundError(f"--skip-export requested but missing {model}")
    else:
        command = [
            python,
            str(SCRIPT_DIRECTORY / "export_mace_mh1_omol.py"),
            "--output",
            str(model),
        ]
        if args.checkpoint is not None:
            command.extend(["--checkpoint", str(args.checkpoint.resolve())])
        export_environment = dict(base_environment)
        export_environment["JAX_PLATFORMS"] = "cuda"
        run_checked(
            "export",
            command,
            environment=export_environment,
            output_directory=output_directory,
        )

    results: list[dict[str, object]] = []
    launcher = shlex.split(args.mpi_launcher)
    for gpu_count, grid in PROCESSOR_GRIDS.items():
        if gpu_count not in gpu_counts:
            continue
        visible_devices = ",".join(devices[:gpu_count])
        for repeat in range(1, args.repeats + 1):
            name = f"lammps_{gpu_count}gpu_repeat{repeat}"
            environment = dict(base_environment)
            environment["CUDA_VISIBLE_DEVICES"] = visible_devices
            command = launcher + [
                "-np",
                str(gpu_count),
                args.lmp,
            ]
            if args.kokkos:
                kokkos_options = [
                        "-k",
                        "on",
                        "g",
                        str(gpu_count),
                        "-sf",
                        "kk",
                        "-pk",
                        "kokkos",
                        "neigh",
                        "half",
                        "newton",
                        "on",
                        "comm",
                        "device",
                        "gpu/aware",
                        "on",
                    ]
                if args.kokkos_binsize is not None:
                    kokkos_options[10:10] = [
                        "binsize",
                        str(args.kokkos_binsize),
                    ]
                command.extend(kokkos_options)
            command.extend(
                [
                    "-in",
                    str(input_file),
                    "-var",
                    "model",
                    str(model),
                    "-var",
                    "data_file",
                    str(data_file),
                    "-var",
                    "warmup_steps",
                    str(args.warmup_steps),
                    "-var",
                    "benchmark_steps",
                    str(args.steps),
                    "-var",
                    "memory_fraction",
                    str(args.memory_fraction),
                    "-var",
                    "atom_capacity",
                    str(args.atom_capacity),
                    "-var",
                    "edge_capacity",
                    str(args.edge_capacity),
                    "-var",
                    "proc_x",
                    str(grid[0]),
                    "-var",
                    "proc_y",
                    str(grid[1]),
                    "-var",
                    "proc_z",
                    str(grid[2]),
                ]
            )
            started = time.perf_counter()
            screen = run_checked(
                name,
                command,
                environment=environment,
                output_directory=output_directory,
            )
            wall_seconds = time.perf_counter() - started
            measurement = parse_measurement(screen, args.steps)
            results.append(
                {
                    "gpu_count": gpu_count,
                    "physical_gpus": visible_devices,
                    "processor_grid": list(grid),
                    "repeat": repeat,
                    "wall_seconds_including_startup": wall_seconds,
                    **measurement,
                }
            )

    summary = {
        "model": str(model),
        "data_file": str(data_file),
        # Only meaningful when prepare actually ran; --skip-prepare reuses an
        # existing data_file that may have come from a different PDB than
        # the (unused) default.
        "pdb": str(args.pdb.resolve()) if not args.skip_prepare else None,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "repeats": args.repeats,
        "memory_fraction": args.memory_fraction,
        "atom_capacity": args.atom_capacity,
        "edge_capacity": args.edge_capacity,
        "kokkos": args.kokkos,
        "kokkos_binsize": args.kokkos_binsize if args.kokkos else None,
        "results": results,
    }
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = sorted({key for result in results for key in result})
    with (output_directory / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {output_directory / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
