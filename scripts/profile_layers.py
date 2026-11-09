"""Per-layer FP32 and INT8 timings on this device, with no network access.

Answers, on the device itself, the two questions the paper's per-layer profiling
is used for:

1. Does each layer run faster in INT8 *here*? Profiles the FP32 model and the
   fully quantized one with the ONNX Runtime profiler and compares every layer
   of the model's sensitivity table. It also says how many layers change side
   (speeds up / does not) with respect to the `dur_s_*` columns of the table
   the JSD + Latency policy was built from.
2. Is a layer left in FP32 between INT8 neighbours slower than in the pure FP32
   model (graph fragmentation)? With `--run-dir`, reads the trace the ramp flow
   already wrote for every configuration and reports, per configuration, the
   time of the depthwise convolutions, of the Q/DQ nodes and of the whole graph.

    uv run python scripts/profile_layers.py experiment=timm_convnext_tiny_susy
    uv run python scripts/profile_layers.py experiment=timm_convnext_tiny_susy \\
        --run-dir outputs/2026-10-01/11-04-12 --write-table

`--write-table` saves the model's sensitivity table with the `dur_s_*` columns
measured here, as `<model>__<device>-reprofiled.csv`, ready to pass to the ramp
flow as `sensitivity_table=`.
"""

import argparse
import logging
import re
import socket
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils
from hydra import compose, initialize_config_dir

PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.quantization.evaluation import (  # noqa: E402
    _model_input_shape,
    calibrate_once,
    get_session_options,
    preprocess_model,
    quantize_nodes,
)
from src.quantization.profiling import QDQ_OPS, node_durations, profile_model  # noqa: E402

TABLES = PROJECT_ROOT / "data" / "paper" / "sensitivity_tables"
# Experiment name -> the model's file prefix in data/paper/sensitivity_tables.
TABLE_NAMES = {
    "timm_resnet18_susy": "resnet18",
    "timm_convnext_tiny_susy": "convnext_tiny",
    "timm_efficientnet-b0_susy": "efficientnet-b0",
    "timm_tinyvit_susy": "tinyvit",
}


def node_times(trace_path: str) -> pd.DataFrame:
    """Median duration (ms) and op type of every node of an ORT trace.

    Node names are cleaned as the paper's analysis did, so a fused or
    quantized node keeps the name of the layer it comes from.
    """
    durations = node_durations(trace_path, quantized=True)
    return pd.DataFrame({"ms": durations["dur_s"] * 1000, "op": durations["op"]})


def kind(node: str) -> str:
    if "conv_dw" in node:
        return "depthwise conv"
    if "/se/" in node:
        return "squeeze-excite conv"
    return node.rsplit("/", 1)[-1]


def profile_device(config, work_dir: str) -> pd.DataFrame:
    """Per-node FP32 and INT8 times of the model, profiled on this machine."""
    options = get_session_options(config.session_options)
    sample = np.random.randn(*_model_input_shape(config.onnx_file)).astype(np.float32)

    fp32_dir, int8_dir = Path(work_dir, "fp32"), Path(work_dir, "int8")
    fp32_dir.mkdir()
    int8_dir.mkdir()
    fp32 = node_times(profile_model(config.onnx_file, str(fp32_dir), sample, options))

    preprocessed = preprocess_model(config.onnx_file, config.quantization, work_dir)
    ranges = calibrate_once(preprocessed, config.quantization, work_dir)
    quantized = str(int8_dir / "quantized_model.onnx")
    quantize_nodes(preprocessed, [], ranges, config.quantization, quantized)
    int8 = node_times(profile_model(quantized, str(int8_dir), sample, options))

    print(f"\nWhole graph, kernel time per run: FP32 {fp32.ms.sum():.1f} ms, uniform INT8 {int8.ms.sum():.1f} ms")
    return pd.DataFrame({"fp32_ms": fp32.ms, "int8_ms": int8.ms}).dropna()


def compare_with_table(times: pd.DataFrame, table: pd.DataFrame, used: pd.DataFrame) -> pd.DataFrame:
    layers = pd.DataFrame({"layer": table["layer"]}).merge(times, left_on="layer", right_index=True, how="left")
    layers["kind"] = layers["layer"].map(kind)
    layers["speeds_up"] = layers["fp32_ms"] > layers["int8_ms"]
    used_speedup = dict(zip(used["layer"], used["dur_s_fp32"] > used["dur_s_int8"]))
    layers["in_table_used"] = layers["layer"].isin(used_speedup)
    layers["speeds_up_in_table_used"] = layers["layer"].map(used_speedup)

    summary = layers.groupby("kind").agg(
        layers=("layer", "size"),
        profiled=("fp32_ms", "count"),
        fp32_ms=("fp32_ms", "sum"),
        int8_ms=("int8_ms", "sum"),
        speed_up_here=("speeds_up", "sum"),
        speed_up_in_table_used=("speeds_up_in_table_used", lambda s: int(s.fillna(False).sum())),
    )
    print("\nLayers of the sensitivity table, by kind:")
    print(summary.round(2).sort_values("fp32_ms", ascending=False).to_string())

    both = layers[layers["in_table_used"] & layers["fp32_ms"].notna()]
    flipped = both[both["speeds_up"] != both["speeds_up_in_table_used"].astype(bool)]
    print(
        f"\n{len(layers)} layers; {int(layers.fp32_ms.notna().sum())} found in both traces; "
        f"{int((~layers.in_table_used).sum())} missing from the table the policy used."
    )
    print(
        f"Of the {len(both)} layers in both, {len(flipped)} change side: "
        f"{int(flipped.speeds_up.sum())} speed up here but not in the table used, "
        f"{int((~flipped.speeds_up).sum())} the other way round."
    )
    if len(flipped):
        print("  by kind:", flipped.groupby("kind").size().to_dict())
    return layers


def mixed_configurations(run_dir: Path, times: pd.DataFrame) -> None:
    """How the FP32 time of a layer changes inside a mixed-precision graph."""
    traces = sorted(run_dir.glob("final_results/*/onnxruntime_profiling.json"))
    if not traces:
        print(f"\nNo traces under {run_dir}/final_results.")
        return
    depthwise = [n for n in times.index if "conv_dw" in n]
    order = {"all_int8": (0, 0)}
    rows = [
        {
            "config": "FP32",
            "depthwise_ms": times.loc[depthwise, "fp32_ms"].sum(),
            "depthwise_in_fp32": len(depthwise),
            "qdq_ms": 0.0,
            "graph_ms": float("nan"),
        }
    ]
    for trace in sorted(traces, key=lambda p: order.get(p.parent.name, (1, p.parent.name))):
        nodes = node_times(str(trace))
        dw = nodes.reindex(depthwise).dropna()
        rows.append(
            {
                "config": trace.parent.name,
                "depthwise_ms": dw.ms.sum(),
                "depthwise_in_fp32": int((dw.op == "Conv").sum()),
                # FP32 depthwise convs only, to compare with the pure FP32 graph.
                "fp32_depthwise_ms": dw.loc[dw.op == "Conv", "ms"].sum(),
                "same_layers_in_fp32_graph_ms": times.reindex(dw.index[dw.op == "Conv"])["fp32_ms"].sum(),
                "qdq_ms": nodes.loc[nodes.op.isin(QDQ_OPS), "ms"].sum(),
                "graph_ms": nodes.ms.sum(),
            }
        )
    print(f"\nConfigurations of {run_dir} (kernel time per run, ms):")
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print(
        "\nA depthwise convolution left in FP32 is slowed down by its INT8 neighbours when "
        "'fp32_depthwise_ms' is clearly above 'same_layers_in_fp32_graph_ms'."
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, help="a ramp run's Hydra output directory")
    parser.add_argument("--device", default=None, help="device tag of the table to compare with (default: cortex-a76)")
    parser.add_argument("--write-table", action="store_true", help="save the table with the timings measured here")
    args, overrides = parser.parse_known_args()

    with initialize_config_dir(str(PROJECT_ROOT / "config"), version_base="1.3"):
        config = compose("ramp.yaml", overrides=["sensitivity_table=unused", *overrides])
    experiment = next((o.split("=", 1)[1] for o in overrides if o.startswith("experiment=")), None)
    model = TABLE_NAMES.get(experiment or "timm_convnext_tiny_susy", None)
    if model is None:
        parser.error(f"No sensitivity table known for experiment '{experiment}'")
    device = args.device or "cortex-a76"
    print(f"{config.model.model_name} on {socket.gethostname()}")

    # The Apple M1 table has every ranked layer and the same sensitivity
    # metrics, which do not depend on the device; only dur_s_* is per device.
    table = pd.read_csv(TABLES / f"{model}__apple-m1.csv")
    used = pd.read_csv(TABLES / f"{model}__{device}.csv")

    with tempfile.TemporaryDirectory() as work_dir:
        times = profile_device(config, work_dir)
    layers = compare_with_table(times, table, used)

    if args.run_dir:
        mixed_configurations(args.run_dir, times)

    if args.write_table:
        measured = table.merge(layers[["layer", "fp32_ms", "int8_ms"]], on="layer").dropna(subset=["fp32_ms", "int8_ms"])
        measured["dur_s_fp32"], measured["dur_s_int8"] = measured.fp32_ms / 1000, measured.int8_ms / 1000
        measured["dur_s_speedup"] = measured.dur_s_fp32 / measured.dur_s_int8
        measured["dur_s_diff"] = measured.dur_s_fp32 - measured.dur_s_int8
        out = TABLES / f"{model}__{re.sub(r'[^a-z0-9]+', '-', device)}-reprofiled.csv"
        measured.drop(columns=["fp32_ms", "int8_ms"]).to_csv(out, index=False)
        print(f"\nWritten {out} ({len(measured)} layers)")


if __name__ == "__main__":
    main()
