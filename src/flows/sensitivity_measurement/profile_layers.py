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

import logging
import os
import socket

import numpy as np
import pandas as pd
from omegaconf import DictConfig

import wandb
from src.quantization.evaluation import (
    _model_input_shape,
    calibrate_once,
    delete_onnx_files,
    get_session_options,
    preprocess_model,
    quantize_nodes,
)
from src.quantization.profiling import node_durations, profile_model

PROFILING_CSV = "layer_profiling.csv"


def analyze_layer_profiling(config: DictConfig, output_dir: str, run: wandb.Run) -> None:
    """Per-node kernel time of the FP32 and the fully quantized model, on this device.

    Writes ``dur_s_fp32`` and ``dur_s_int8`` for every node present in both ORT
    profiling traces: the per-layer latency the ``ramp`` flow uses to decide
    whether quantizing a layer pays off (``final_results.latency_measure: dur_s``).
    The timings belong to the machine this runs on.
    """
    wandb.run.tags = wandb.run.tags + ("profile-layers",)
    onnx_file = config.get("onnx_file")
    quantization_config = config.get("quantization")
    session_options = get_session_options(config.get("session_options", {}))

    work_dir = os.path.join(output_dir, "profile_layers")
    fp32_dir, int8_dir = os.path.join(work_dir, "fp32"), os.path.join(work_dir, "int8")
    os.makedirs(fp32_dir, exist_ok=True)
    os.makedirs(int8_dir, exist_ok=True)

    # Random input of the model's shape, as the paper's profiling did.
    sample = np.random.randn(*_model_input_shape(onnx_file)).astype(np.float32)

    # FP32 on the original file, as the paper's base run (never preprocessed).
    fp32_trace = profile_model(onnx_file, fp32_dir, sample, session_options)

    preprocessed = preprocess_model(onnx_file, quantization_config, work_dir)
    tensors_range = calibrate_once(preprocessed, quantization_config, work_dir)
    quantized = os.path.join(int8_dir, "quantized_model.onnx")
    # An empty node list quantizes every node the configured op types allow.
    quantize_nodes(preprocessed, [], tensors_range, quantization_config, quantized)
    int8_trace = profile_model(quantized, int8_dir, sample, session_options)

    fp32 = node_durations(fp32_trace, quantized=False)
    int8 = node_durations(int8_trace, quantized=True)
    df = pd.DataFrame({"dur_s_fp32": fp32["dur_s"], "dur_s_int8": int8["dur_s"]}).dropna()
    df["dur_s_speedup"] = df["dur_s_fp32"] / df["dur_s_int8"]
    df["dur_s_diff"] = df["dur_s_fp32"] - df["dur_s_int8"]
    df = df.rename_axis("node").reset_index()
    logging.info(
        "Layer profiling: %d nodes in FP32, %d in INT8, %d in both; %d of them "
        "run faster in INT8.",
        len(fp32),
        len(int8),
        len(df),
        int((df["dur_s_speedup"] > 1).sum()),
    )

    csv_path = os.path.join(output_dir, PROFILING_CSV)
    df.to_csv(csv_path, index=False)

    run.log({"profiling/layer_table": wandb.Table(dataframe=df)})
    artifact = wandb.Artifact(
        name="layer-profiling",
        type="analysis",
        metadata={"host": socket.gethostname(), "num_nodes": str(len(df))},
    )
    artifact.add_file(csv_path)
    artifact.add_file(fp32_trace, name="fp32/onnxruntime_profiling.json")
    artifact.add_file(int8_trace, name="int8/onnxruntime_profiling.json")
    run.log_artifact(artifact)

    delete_onnx_files(work_dir)
