#!/usr/bin/env python3
"""Export MACE-MH-1 with the OMOL head as a CUDA chemtrain bundle."""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path


# These must be set before importing JAX, MACE, or OpenEquivariance. In
# particular, disabling TF32 keeps the exported fp32 matmul contract
# reproducible across benchmark runs.
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "float32"
os.environ["OEQ_NOTORCH"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.getuid()}")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["JAX_PLATFORMS"] = "cuda"


SPECORDER = ("H", "C", "N", "O", "Na", "P", "Cl")
ATOMIC_NUMBERS = (1, 6, 7, 8, 11, 15, 17)


def _load_model(compose, checkpoint: Path | None, family: str, version: str):
    """Load a checkpoint without pruning the multi-head foundation model."""
    if checkpoint is None:
        configured = os.environ.get("MACE_MH1_CHECKPOINT")
        if configured:
            checkpoint = Path(configured).expanduser()
        else:
            cached = Path.home() / ".cache" / "mace" / "macemh1model"
            if cached.is_file():
                checkpoint = cached

    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"MACE checkpoint not found: {checkpoint}")
        return compose.load_torch_model(str(checkpoint))

    return compose.load_foundational_model(family=family, version=version)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Original multi-head checkpoint; defaults to MACE_MH1_CHECKPOINT or cache.",
    )
    parser.add_argument("--family", default="mp")
    parser.add_argument("--version", default="mh-1")
    args = parser.parse_args()

    import jax.numpy as jnp
    from chemtrain.compose import mace_jax as mace_compose
    from chemtrain.deploy import exporter, graphs
    from jax_md import space
    from mace_jax.modules.wrapper_ops import EquivarianceConfig

    torch_model, model_config = _load_model(
        mace_compose,
        args.checkpoint,
        args.family,
        args.version,
    )
    heads = tuple(str(head) for head in (getattr(torch_model, "heads", ()) or ()))
    if "omol" not in heads:
        raise ValueError(
            "The original MACE-MH-1 checkpoint does not expose the OMOL head; "
            f"available heads are {heads!r}."
        )

    class MaceDnaExporter(exporter.Exporter):
        """Map compact LAMMPS atom types to MACE atomic numbers."""

        graph_type = graphs.SimpleSparseNeighborList
        unit_style = "metal"

        def __init__(self, model_apply, config):
            self.model = model_apply
            self.r_cutoff = float(config["r_max"])
            interactions = int(config["num_interactions"])
            self.nbr_order = [interactions, 2 * interactions]
            super().__init__()

        def energy_fn(self, position, particle_data, graph, comm=None):
            neighbor = graph.to_neighborlist()
            atomic_numbers = jnp.asarray(ATOMIC_NUMBERS, dtype=jnp.int32)
            species = atomic_numbers[particle_data["species"]]
            return self.model(
                position,
                neighbor,
                species=species,
                comm=comm,
            )

    equivariance = EquivarianceConfig(
        backend="openeq",
        layout="mul_ir",
        group="O3_e3nn",
        optimize_channelwise=True,
        conv_fusion=True,
    )
    displacement, _ = space.free()
    variables, apply_fn = mace_compose.mace_jax_neighborlist_from_torch(
        model_config,
        torch_model,
        displacement,
        max_edge_multiplier=None,
        per_particle=True,
        species_mapping=mace_compose.AtomicNumberMapping(max_number=100),
        scale_pot=1.0,
        scale_pos=1.0,
        equivariance_config=equivariance,
        head="omol",
    )
    model = MaceDnaExporter(functools.partial(apply_fn, variables), model_config)
    model.export(
        communication=True,
        custom_calls=exporter.OPENEQUIVARIANCE_CUSTOM_CALLS,
        platforms=("cuda",),
    )

    variants = {variant.name: variant for variant in model._proto.variants}
    expected = {
        "comm_off_newton_off",
        "comm_off_newton_on",
        "comm_on_newton_on",
    }
    if set(variants) != expected:
        raise RuntimeError(
            f"Expected variants {sorted(expected)}, got {sorted(variants)}"
        )
    if variants["comm_on_newton_on"].communication_buffer_width <= 0:
        raise RuntimeError("The communication-enabled OMOL variant has no buffer")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(
        json.dumps(
            {
                "family": args.family,
                "version": args.version,
                "head": "omol",
                "heads": list(heads),
                "atomic_numbers": list(ATOMIC_NUMBERS),
                "lammps_specorder": list(SPECORDER),
                "r_max_angstrom": float(model_config["r_max"]),
                "num_interactions": int(model_config["num_interactions"]),
                "variants": sorted(variants),
                "communication_buffer_width": variants[
                    "comm_on_newton_on"
                ].communication_buffer_width,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Exported MACE-MH-1 head=omol, cutoff={model_config['r_max']}, "
        f"variants={sorted(variants)}, output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
