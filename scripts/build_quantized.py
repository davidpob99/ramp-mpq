"""Build a quantized ONNX model from the FP32 model and quantization_policies.json.

The JSON written by the ramp flow gives, for every configuration of the sweep,
the precision of every quantizable node; the ones in the paper's table are
marked in ``selected_for``. This script rebuilds any of them exactly as the flow
did: quant_pre_process, one MinMax calibration pass (the seeded image subset of
the calibration folder) and QDQ quantization of the int8 nodes.

    # The configuration the elbow picked for a policy
    uv run python scripts/build_quantized.py quantization_policies.json --policy "RAMP (JSD)"
    # Any configuration of the sweep, by name
    uv run python scripts/build_quantized.py quantization_policies.json --config jsd_2
    # List what the JSON holds
    uv run python scripts/build_quantized.py quantization_policies.json --list

Which layers are int8 is fixed by the JSON. The scales come from calibration,
so the model matches the original one exactly only with the same calibration
images (same SuSy train folder, 1000 images, seed 42) and ONNX Runtime version.
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import rootutils
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict

PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.quantization.evaluation import (  # noqa: E402
    calibrate_once,
    preprocess_model,
    quantize_nodes,
)

CONFIG_DIR = PROJECT_ROOT / "config"


def _compose(experiment: str, overrides: List[str]) -> DictConfig:
    with initialize_config_dir(str(CONFIG_DIR), version_base="1.3"):
        return compose(
            "ramp.yaml",
            overrides=[f"experiment={experiment}", "sensitivity_table=unused", *overrides],
        )


def _experiment_for(model_name: str) -> str:
    """The experiment whose model is ``model_name``: it carries the quantization
    scope (e.g. TinyViT only quantizes Gemm, MatMul and Conv)."""
    for path in sorted((CONFIG_DIR / "experiment").glob("*.yaml")):
        if _compose(path.stem, []).model.get("model_name") == model_name:
            return path.stem
    sys.exit(f"No experiment in config/experiment for model '{model_name}'; pass --experiment.")


def _pick(policies: List[dict], config: Optional[str], policy: Optional[str]) -> dict:
    if config:
        matches = [p for p in policies if p["name"] == config]
    else:
        matches = [p for p in policies if p.get("selected_for") == policy]
    if not matches:
        wanted = config or policy
        sys.exit(f"'{wanted}' is not in the JSON; see --list.")
    return matches[0]


def _list(data: dict) -> None:
    print(f"{data['model_name']}: {data['quantizable_nodes']} quantizable nodes\n")
    # Published policy files carry no measurements; the columns are then left out.
    measured = any(p.get("results") for p in data["policies"])
    header = f"{'config':18s}{'int8':>6s}"
    if measured:
        header += f"{'acc':>8s}{'latency ms':>12s}"
    print(f"{header}  selected for")
    seen = set()
    for p in data["policies"]:
        # all_int8 opens every metric's sweep; it is the same model each time.
        if p["name"] in seen:
            continue
        seen.add(p["name"])
        row = f"{p['name']:18s}{p.get('n_int8', 0):>6d}"
        if measured:
            results = p.get("results") or {}
            acc = results.get("acc")
            latency = results.get("latency")
            row += (
                f"{acc * 100 if acc is not None else float('nan'):>8.2f}"
                f"{latency * 1000 if latency is not None else float('nan'):>12.2f}"
            )
        print(f"{row}  {p.get('selected_for') or ''}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("policies", type=Path, help="quantization_policies.json")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--policy", help="a policy of the paper's table, e.g. 'RAMP (JSD)'")
    which.add_argument("--config", help="a configuration of the sweep, e.g. jsd_2")
    which.add_argument("--list", action="store_true", help="list the configurations")
    parser.add_argument("--onnx", type=Path, help="FP32 model (default: the experiment's)")
    parser.add_argument("--output", type=Path, help="default: <model>__<config>.onnx")
    parser.add_argument("--experiment", help="default: found from the JSON's model name")
    parser.add_argument("--calibration-dir", type=Path, help="default: the config's")
    parser.add_argument("--max-images", type=int, help="default: the config's (1000)")
    args = parser.parse_args()

    data = json.loads(args.policies.read_text())
    if args.list or not (args.config or args.policy):
        _list(data)
        return

    chosen = _pick(data["policies"], args.config, args.policy)
    if chosen["name"] == "base":
        sys.exit("'base' is the FP32 model itself; nothing to quantize.")

    experiment = args.experiment or _experiment_for(data["model_name"])
    config = _compose(experiment, [])
    quantization = config.quantization
    with open_dict(quantization):
        if args.calibration_dir:
            quantization.calibration_data.folder_path = str(args.calibration_dir)
        if args.max_images:
            quantization.calibration_data.max_images = args.max_images
    onnx_file = str(args.onnx or config.onnx_file)

    # all_int8 was built with an empty node list, which lets ONNX Runtime
    # quantize every node of the configured op types; the rest name their nodes.
    if chosen["name"] == "all_int8":
        nodes: List[str] = []
    else:
        nodes = [node for node, precision in chosen["layers"].items() if precision == "int8"]
    output = args.output or Path(f"{data['model_name']}__{chosen['name']}.onnx")
    logging.info(
        "%s: %s (%s), %d of %d nodes int8, experiment %s",
        data["model_name"], chosen["name"], chosen.get("selected_for") or "-",
        chosen.get("n_int8", len(nodes)), data["quantizable_nodes"], experiment,
    )

    with tempfile.TemporaryDirectory() as work_dir:
        preprocessed = preprocess_model(onnx_file, quantization, work_dir)
        tensors_range = calibrate_once(preprocessed, quantization, work_dir)
        quantize_nodes(preprocessed, nodes, tensors_range, quantization, str(output))
    logging.info("Written %s", output)


if __name__ == "__main__":
    main()
