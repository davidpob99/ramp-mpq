#  Copyright 2026 David Población Criado
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import json
import logging
import os
import re
import shutil
from typing import Any, Dict, List, Set

import numpy as np
import onnxruntime as ort
import pandas as pd

QDQ_OPS = ("QuantizeLinear", "DequantizeLinear")


def profile_model(
        model_path: str,
        output_dir: str,
        sample_input: np.ndarray,
        session_options: ort.SessionOptions,
        warmup_runs: int = 5,
        num_runs: int = 50,
) -> str:
    """Run the ORT profiler and return the path of the trace it writes."""
    profiling_options = ort.SessionOptions()
    profiling_options.graph_optimization_level = session_options.graph_optimization_level
    profiling_options.enable_profiling = True

    session = ort.InferenceSession(model_path, profiling_options)
    output_names = [output.name for output in session.get_outputs()]
    for _ in range(warmup_runs + num_runs):
        session.run(output_names, {"input": sample_input})

    trace_path = session.end_profiling()
    del session

    destination = os.path.join(output_dir, "onnxruntime_profiling.json")
    shutil.move(trace_path, destination)
    return destination


def _layer_durations(profiling_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """One row per profiled node execution, with its op type and duration."""
    rows = [
        {
            "layer_name": event["name"].replace("_kernel_time", ""),
            "op_name": event["args"]["op_name"],
            "duration_ms": event["dur"] / 1000.0,
        }
        for event in profiling_data
        if event.get("cat") == "Node" and "name" in event and "dur" in event
    ]
    return pd.DataFrame(rows, columns=["layer_name", "op_name", "duration_ms"])


def measure_qdq_overhead(trace_path: str) -> Dict[str, float]:
    """Aggregate the time spent in Q/DQ nodes, as absolute ms and as a share"""
    empty = {
        "qdq_total_ms": float("nan"),
        "qdq_num_layers": float("nan"),
        "qdq_pct": float("nan"),
        "profiled_total_ms": float("nan"),
        "profiled_num_layers": float("nan"),
    }
    try:
        with open(trace_path) as handle:
            profiling_data = json.load(handle)
    except (OSError, ValueError) as exc:
        logging.warning("Could not read the ORT trace at %s: %s", trace_path, exc)
        return empty

    durations = _layer_durations(profiling_data)
    if durations.empty:
        logging.warning("The ORT trace at %s has no node events", trace_path)
        return empty

    # Median across repetitions first, so one slow run does not dominate.
    per_layer = durations.groupby("layer_name").agg(
        duration_ms=("duration_ms", "median"), op_name=("op_name", "first")
    )
    is_qdq = per_layer["op_name"].isin(QDQ_OPS)
    qdq_total = float(per_layer.loc[is_qdq, "duration_ms"].sum())
    profiled_total = float(per_layer["duration_ms"].sum())

    return {
        "qdq_total_ms": qdq_total,
        "qdq_num_layers": float(is_qdq.sum()),
        "qdq_pct": 100.0 * qdq_total / profiled_total if profiled_total > 0 else float("nan"),
        "profiled_total_ms": profiled_total,
        "profiled_num_layers": float(len(per_layer)),
    }


def _clean_node_name(name: str, quantized: bool) -> str:
    """The ONNX node a trace event belongs to, as the paper's analysis named it.

    A fused or quantized node keeps the name of the layer it comes from.
    """
    name = name.replace("_kernel_time", "")
    if quantized:
        name = re.sub(r"_token_\d+", "", name)
    return re.sub(r"_output_\d+_\w+", "", name)


def node_durations(trace_path: str, quantized: bool) -> pd.DataFrame:
    """Median kernel time (``dur_s``, seconds) and op type of every node of a trace."""
    with open(trace_path) as handle:
        events = json.load(handle)
    rows = [
        {
            "node": _clean_node_name(event["name"], quantized),
            "dur_s": event["dur"] / 1e6,
            "op": event.get("args", {}).get("op_name"),
        }
        for event in events
        if event.get("cat") == "Node"
        and isinstance(event.get("name"), str)
        and event["name"].endswith("_kernel_time")
    ]
    frame = pd.DataFrame(rows, columns=["node", "dur_s", "op"])
    # Median across repetitions, so one slow run does not dominate.
    return frame.groupby("node").agg(dur_s=("dur_s", "median"), op=("op", "first"))


def profiled_node_names(trace_path: str, quantized: bool) -> Set[str]:
    """ONNX node names that appear in an ORT profiling trace

    A node missing from a trace was fused away by graph optimization, so it has
    no runtime of its own on that hardware
    """
    with open(trace_path) as handle:
        events = json.load(handle)

    names = set()
    for event in events:
        name = event.get("name")
        if not isinstance(name, str):
            continue
        name = name.replace("_kernel_time", "")
        if quantized:
            name = re.sub(r"_token_\d+", "", name)
        names.add(re.sub(r"_output_\d+_\w+", "", name))
    return names


def profiled_node_intersection(fp32_trace: str, int8_trace: str) -> Set[str]:
    """Nodes present in both traces"""
    fp32 = profiled_node_names(fp32_trace, quantized=False)
    int8 = profiled_node_names(int8_trace, quantized=True)
    logging.info(
        "Profiling traces: %d nodes in FP32, %d in INT8, %d in both",
        len(fp32),
        len(int8),
        len(fp32 & int8),
    )
    return fp32 & int8
