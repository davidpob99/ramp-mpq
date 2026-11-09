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
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import seaborn as sns
import torch
import wandb
from omegaconf import DictConfig
from torchmetrics import (
    CosineSimilarity,
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
    NormalizedRootMeanSquaredError,
)
from torchmetrics.classification import MulticlassAccuracy
from tqdm.auto import tqdm

from src.metrics.divergence import compute_batch_jsd_mean, compute_batch_kl_mean
from src.metrics.signal import compute_batch_snr_mean
from src.quantization.evaluation import (
    EvaluationContext,
    build_evaluation_context,
    get_quantizable_nodes,
    measure_latency,
    quantize_nodes,
    to_2d_float_tensor,
)

_OAT_METRIC_KEYS = (
    "mse",
    "mae",
    "nrmse",
    "cos_dis",
    "kl",
    "snr",
    "jsd",
    "acc_fp32",
    "acc_int8",
    "acc_drop",
    "latency_fp32",
    "latency_int8",
    "speedup",
)


def _oat_result(**overrides: float) -> Dict[str, float]:
    """Build an OAT per-layer result row with a fixed schema (NaN by default)"""
    row = {key: float("nan") for key in _OAT_METRIC_KEYS}
    row.update(overrides)
    return row


def _compare_oat_outputs(
        session_int8: ort.InferenceSession,
        output_names: List[str],
        batch_inputs: List[np.ndarray],
        batch_outputs_fp32: List[np.ndarray],
        batch_labels: List[np.ndarray],
        num_classes: int,
        layer_name: str,
) -> Optional[Dict[str, float]]:
    """Output-drift metrics between FP32 and the partially-quantized model."""
    # NRMSE with normalization="std" so the error is relative to the
    # reference tensor's spread, independent of its absolute scale.
    # This keeps layers with larger-norm outputs from looking falsely
    # less sensitive.
    metrics = MetricCollection(
        {
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "nrmse": NormalizedRootMeanSquaredError(normalization="std"),
            "cos_sim": CosineSimilarity(reduction="mean"),
            "acc": MulticlassAccuracy(num_classes=num_classes, average="micro"),
        }
    )
    kl_sum = 0.0
    snr_sum = 0.0
    jsd_sum = 0.0
    n_batches = 0

    for images, out_fp32, labels in zip(batch_inputs, batch_outputs_fp32, batch_labels):
        out_int8 = session_int8.run(output_names, {"input": images})[0]

        if out_fp32.shape != out_int8.shape:
            logging.warning(
                "Shape mismatch for node '%s': fp32=%s int8=%s — skipping batch",
                layer_name,
                out_fp32.shape,
                out_int8.shape,
            )
            continue

        fp32_t = to_2d_float_tensor(out_fp32)
        int8_t = to_2d_float_tensor(out_int8)

        metrics["mse"].update(int8_t, fp32_t)
        metrics["mae"].update(int8_t, fp32_t)
        metrics["nrmse"].update(int8_t, fp32_t)
        metrics["cos_sim"].update(int8_t, fp32_t)
        # Accuracy over the same batches that survive the shape filter, so the
        # denominator matches the one used for the drift metrics.
        metrics["acc"].update(int8_t, torch.from_numpy(labels))

        # _compute_batch_kl_mean expects raw logits and applies
        # log_softmax / softmax internally.
        kl_sum += compute_batch_kl_mean(
            prediction_logits=int8_t,
            target_logits=fp32_t,
        )
        snr_sum += compute_batch_snr_mean(
            reference=fp32_t,
            estimate=int8_t,
        )
        jsd_sum += compute_batch_jsd_mean(
            prediction_logits=int8_t,
            target_logits=fp32_t,
        )
        n_batches += 1

        del fp32_t, int8_t

    if n_batches == 0:
        return None
    return {
        "mse": float(metrics["mse"].compute().item()),
        "mae": float(metrics["mae"].compute().item()),
        "nrmse": float(metrics["nrmse"].compute().item()),
        "cos_dis": 1 - float(metrics["cos_sim"].compute().item()),
        "kl": kl_sum / n_batches,
        "snr": snr_sum / n_batches,
        "jsd": jsd_sum / n_batches,
        "acc_int8": float(metrics["acc"].compute().item()),
    }


@dataclass(frozen=True)
class _OatContext:
    """The shared evaluation state plus the OAT loop's own working directory"""

    evaluation: EvaluationContext
    oat_work_dir: str
    keep_node_models: bool = False


def _evaluate_oat_node(layer_name: str, ctx: _OatContext) -> Dict[str, float]:
    """Quantize ``layer_name`` alone and build its OAT result row"""
    layer_oat_dir = os.path.join(ctx.oat_work_dir, f"node_{layer_name}")
    os.makedirs(layer_oat_dir, exist_ok=True)
    try:
        return _measure_oat_node(
            layer_name,
            os.path.join(layer_oat_dir, "quantized_model.onnx"),
            ctx.evaluation,
        )
    finally:
        if not ctx.keep_node_models:
            shutil.rmtree(layer_oat_dir, ignore_errors=True)


def _measure_oat_node(
        layer_name: str, model_int8_path: str, evaluation: EvaluationContext
) -> Dict[str, float]:
    """Quantize one node into ``model_int8_path`` and compare it to FP32."""
    try:
        quantize_nodes(
            model_path=evaluation.preprocessed_path,
            node_names=[layer_name],
            tensors_range=evaluation.tensors_range,
            config_quantization=evaluation.quantization_config,
            output_path=model_int8_path,
        )
    except Exception as exc:
        logging.warning("Error quantizing node '%s': %s", layer_name, exc)
        return _oat_result()

    session_int8 = ort.InferenceSession(model_int8_path, evaluation.session_options)
    latency_int8 = measure_latency(
        model_int8_path,
        evaluation.batch_inputs[0],
        evaluation.session_options,
        warmup_runs=evaluation.warmup_runs,
    )

    drift = _compare_oat_outputs(
        session_int8,
        evaluation.output_names,
        evaluation.batch_inputs,
        evaluation.batch_outputs_fp32,
        evaluation.batch_labels,
        evaluation.num_classes,
        layer_name,
    )
    # Release the session before the caller deletes the model file under it.
    del session_int8
    if drift is None:
        logging.warning("No batches were compared for node '%s'", layer_name)
        return _oat_result(
            acc_fp32=evaluation.acc_fp32,
            latency_fp32=evaluation.latency_fp32,
            latency_int8=latency_int8,
        )
    return _oat_result(
        **drift,
        acc_fp32=evaluation.acc_fp32,
        acc_drop=evaluation.acc_fp32 - drift["acc_int8"],
        latency_fp32=evaluation.latency_fp32,
        latency_int8=latency_int8,
        speedup=evaluation.latency_fp32 / latency_int8,
    )


def _write_oat_report(
        results: Dict[str, Dict[str, float]],
        output_dir: str,
        run: wandb.Run,
        onnx_file: str,
        num_batches: int,
        random_seed: int,
) -> None:
    """Save the OAT results CSV and ranking plot and log both to W&B"""
    df_results = (
        pd.DataFrame(results)
        .transpose()
        .reset_index()
        .rename(columns={"index": "layer"})
    )
    df_results = df_results.sort_values("nrmse", ascending=False).reset_index(drop=True)

    results_csv_path = os.path.join(output_dir, "oat_sensitivity_results.csv")
    df_results.to_csv(results_csv_path, index=False)

    ranking_json_path = os.path.join(output_dir, "sensitivity_analysis_results.json")
    df_results.rename(
        columns={
            "layer": "label",
            "acc_int8": "acc",
            "latency_int8": "time_inference_median",
        }
    )[["label", "acc", "time_inference_median"]].to_json(
        ranking_json_path, orient="split"
    )

    run.log({"oat/results": wandb.Table(dataframe=df_results)})

    # --- Plot ---
    df_melted = df_results.melt(id_vars="layer", var_name="metric", value_name="value")
    g = sns.catplot(
        data=df_melted,
        x="layer",
        y="value",
        col="metric",
        kind="bar",
        sharey=False,
        sharex=False,
        height=5,
        aspect=2,
        col_wrap=2,
    )
    g.set_xticklabels(rotation=90)
    g.set_axis_labels("Layer", "Value")
    g.set_titles("Metric: {col_name}")
    g.figure.tight_layout()

    plot_path = os.path.join(output_dir, "oat_sensitivity_ranking.png")
    plt.savefig(plot_path)
    plt.close(g.figure)
    run.log({"oat/ranking_plot": wandb.Image(plot_path)})

    artifact = wandb.Artifact(
        name=f"oat-sensitivity-{run.id}",
        type="analysis",
        metadata={
            "model_path": onnx_file,
            "analysis_type": "oat",
            "num_batches": str(num_batches),
            "num_nodes": str(len(df_results)),
            "random_seed": str(random_seed),
        },
    )
    artifact.add_file(results_csv_path)
    artifact.add_file(ranking_json_path)
    artifact.add_file(plot_path)
    run.log_artifact(artifact)

    logging.info(
        "OAT completed. %d nodes analyzed over %d batches.",
        len(df_results),
        num_batches,
    )


def analyze_oat(config: DictConfig, output_dir: str, run: wandb.Run) -> None:
    """
    OAT (One-At-a-Time) sensitivity analysis.

    For each layer in `layers_to_analyze`:
      - Quantizes only that layer to INT8 (rest stays FP32).
      - Measures the drift between the preprocessed FP32 baseline and the
        partially-quantized model on the final model output.

    The preprocessed FP32 model is used as the baseline so that the only variable
    between iterations is which node is quantized. Calibration uses a fixed seed
    and a fixed subset to ensure comparability across nodes.
    """
    wandb.run.tags = wandb.run.tags + ("analyze-oat",)
    onnx_file = config.get("onnx_file")
    random_seed = int(config.get("seed", 42))

    oat_work_dir = os.path.join(output_dir, "oat")
    evaluation = build_evaluation_context(config, oat_work_dir)

    layers_to_analyze = config.get("layers_to_analyze", [])
    if not layers_to_analyze:
        logging.info(
            "'layers_to_analyze' not set — discovering quantizable nodes in the model."
        )
        layers_to_analyze = get_quantizable_nodes(
            onnx_file, evaluation.quantization_config
        )

    logging.info("Starting OAT analysis for %d layers.", len(layers_to_analyze))
    context = _OatContext(
        evaluation=evaluation,
        oat_work_dir=oat_work_dir,
        keep_node_models=bool(config.get("keep_node_models", False)),
    )
    results: Dict[str, Dict[str, float]] = {}
    for layer_name in tqdm(layers_to_analyze, desc="OAT Sensitivity Analysis"):
        results[layer_name] = _evaluate_oat_node(layer_name, context)

    _write_oat_report(
        results,
        output_dir,
        run,
        onnx_file,
        len(evaluation.batch_inputs),
        random_seed,
    )
