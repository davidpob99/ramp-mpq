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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import numpy as np
import onnx
import onnxruntime as ort
import torch
from omegaconf import DictConfig
from onnxruntime.quantization import (
    QuantFormat,
    QuantizationMode,
    QuantType,
    quant_pre_process,
)
from onnxruntime.quantization.calibrate import (
    CalibrationMethod,
    TensorsData,
    create_calibrator,
)
from onnxruntime.quantization.quant_utils import load_model_with_shape_infer
from onnxruntime.quantization.quantize import ONNXQuantizer, QDQQuantizer
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    F1Score,
    MulticlassAccuracy,
    Precision,
    Recall,
)
from tqdm.auto import tqdm

from src.quantization.onnx import (
    SUPPORTED_QUANT_OP_TYPES,
    CalibrationDataReaderFromFolder,
)
from src.quantization.profiling import measure_qdq_overhead, profile_model


def cfg_get(config, key, default=None):
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def get_session_options(config_sess_opts) -> ort.SessionOptions:
    """Build an ``ort.SessionOptions`` from the ``session_options`` config block."""
    session_options = ort.SessionOptions()
    # Only set when configured, so flows that do not declare them keep ORT's
    # defaults. The paper's evaluations ran with 8 intra-op and 1 inter-op thread.
    if config_sess_opts.get("intra_op_num_threads") is not None:
        session_options.intra_op_num_threads = int(config_sess_opts.intra_op_num_threads)
    if config_sess_opts.get("inter_op_num_threads") is not None:
        session_options.inter_op_num_threads = int(config_sess_opts.inter_op_num_threads)
    if config_sess_opts.get("graph_optimization_level"):
        session_options.graph_optimization_level = getattr(
            ort.GraphOptimizationLevel, config_sess_opts.graph_optimization_level
        )
    if config_sess_opts.get("optimized_model_filepath"):
        session_options.optimized_model_filepath = (
            config_sess_opts.optimized_model_filepath
        )
    if config_sess_opts.get("intra_op_thread_affinities"):
        session_options.add_session_config_entry(
            "session.intra_op_thread_affinities",
            config_sess_opts.get("intra_op_thread_affinities"),
        )
    if config_sess_opts.get("inter_op_thread_affinities"):
        session_options.add_session_config_entry(
            "session.inter_op_thread_affinities",
            config_sess_opts.get("inter_op_thread_affinities"),
        )
    return session_options


def to_2d_float_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a 2D float tensor without unnecessary copies."""
    tensor = torch.from_numpy(array)
    if tensor.ndim == 0:
        return tensor.unsqueeze(0).unsqueeze(0).float()
    if tensor.ndim == 1:
        return tensor.unsqueeze(0).float()
    return tensor.flatten(1).float()


def measure_latency(
        model_path: str,
        sample_input: np.ndarray,
        session_options: ort.SessionOptions,
        n_runs: int = 500,
        warmup_runs: int = 10,
) -> float:
    """
    Measure end-to-end median inference latency in seconds

    Uses a fixed static input to isolate pure inference time from
    dataloader overhead. Reports the median over n_runs to suppress
    OS scheduling jitter (long-tail latency distribution)
    """
    session = ort.InferenceSession(model_path, session_options)
    output_names = [o.name for o in session.get_outputs()]

    for _ in range(warmup_runs):
        session.run(output_names, {"input": sample_input})

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(output_names, {"input": sample_input})
        latencies.append(time.perf_counter() - start)

    return float(np.median(latencies))


def calibrate_once(
        model_path: str,
        config_quantization: Dict,
        output_dir: str,
) -> TensorsData:
    """
    Run calibration once on the full model and return the TensorsData ranges.
    These ranges are reused for every OAT node quantization, avoiding N
    redundant calibration passes.
    """
    calibration_data = cfg_get(config_quantization, "calibration_data")
    calib_reader = CalibrationDataReaderFromFolder(
        folder_path=cfg_get(calibration_data, "folder_path"),
        size=tuple(cfg_get(calibration_data, "size", ())),
        input_name=cfg_get(calibration_data, "input_name", "input"),
        max_images=int(cfg_get(calibration_data, "max_images", 1000)),
        seed=int(cfg_get(calibration_data, "seed", 42)),
    )
    augmented_path = os.path.join(output_dir, "augmented_model.onnx")
    calibrator = create_calibrator(
        model=model_path,
        op_types_to_calibrate=None,
        augmented_model_path=augmented_path,
        calibrate_method=CalibrationMethod.MinMax,
    )
    calibrator.collect_data(calib_reader)
    tensors_range = calibrator.compute_data()
    if not isinstance(tensors_range, TensorsData):
        raise TypeError(
            f"Unexpected type from calibrator.compute_data(): {type(tensors_range)}"
        )
    logging.info(
        "Calibration completed. %d tensor ranges collected.",
        len(list(tensors_range.keys())),
    )
    for tensor_name, tensor_data in tensors_range.items():
        # Extract min/max depending on the ONNX Runtime API version
        if hasattr(tensor_data, "range_value"):
            rmin, rmax = tensor_data.range_value
        elif hasattr(tensor_data, "lowest") and hasattr(tensor_data, "highest"):
            rmin, rmax = tensor_data.lowest, tensor_data.highest
        else:
            # Fallback for older versions where this was a plain tuple
            rmin, rmax = tensor_data

        if rmin >= rmax:
            logging.warning(
                "Anomalous calibration range detected for '%s': min=%f, max=%f",
                tensor_name,
                rmin,
                rmax,
            )
    return tensors_range


def _scope_label(node_names: List[str]) -> str:
    """Human-readable name for a quantization scope, for log messages."""
    if not node_names:
        return "the whole model"
    if len(node_names) == 1:
        return f"node '{node_names[0]}'"
    return f"{len(node_names)} nodes"


def quantize_nodes(
        model_path: str,
        node_names: List[str],
        tensors_range: TensorsData,
        config_quantization: Dict,
        output_path: str,
) -> str:
    """Quantize the given nodes reusing precomputed calibration ranges"""
    model_path_obj = Path(model_path)
    output_path_obj = Path(output_path)

    quantization_args = hydra.utils.instantiate(
        cfg_get(config_quantization, "quantization_args", {})
    )
    quant_format = quantization_args.pop("quant_format", QuantFormat.QDQ)
    weight_type = quantization_args.pop("weight_type", QuantType.QInt8)
    activation_type = quantization_args.pop("activation_type", QuantType.QInt8)
    per_channel = quantization_args.pop("per_channel", False)
    reduce_range = quantization_args.pop("reduce_range", False)
    op_types_to_quantize = quantization_args.pop("op_types_to_quantize", [])
    if not op_types_to_quantize:
        op_types_to_quantize = list(SUPPORTED_QUANT_OP_TYPES)

    model = load_model_with_shape_infer(model_path_obj)

    if quant_format == QuantFormat.QDQ:
        quantizer = QDQQuantizer(
            model=model,
            per_channel=per_channel,
            reduce_range=reduce_range,
            weight_qType=weight_type,
            activation_qType=activation_type,
            tensors_range=tensors_range,
            nodes_to_quantize=list(node_names),
            nodes_to_exclude=[],
            op_types_to_quantize=op_types_to_quantize,
            extra_options=quantization_args,
        )
    else:
        quantizer = ONNXQuantizer(
            model=model,
            per_channel=per_channel,
            reduce_range=reduce_range,
            mode=QuantizationMode.QLinearOps,
            static=True,
            weight_qType=weight_type,
            activation_qType=activation_type,
            tensors_range=tensors_range,
            nodes_to_quantize=list(node_names),
            nodes_to_exclude=[],
            op_types_to_quantize=op_types_to_quantize,
            extra_options=quantization_args,
        )

    quantizer.quantize_model()
    quantizer.model.save_model_to_file(output_path_obj, use_external_data_format=False)
    try:
        onnx.checker.check_model(output_path_obj)
        q_model = onnx.load(output_path_obj)
        has_quant_nodes = any(
            "Quantize" in node.op_type or "QLinear" in node.op_type
            for node in q_model.graph.node
        )

        if not has_quant_nodes:
            logging.error(
                "Silent failure! The model generated for %s contains no INT8 ops "
                "(still pure FP32).",
                _scope_label(node_names),
            )

    except onnx.checker.ValidationError as e:
        logging.error(
            "Invalid ONNX graph generated for %s: %s", _scope_label(node_names), e
        )
        raise ValueError(f"The quantized model is corrupt: {e}")
    return output_path


def get_quantizable_nodes(
        model_path: str,
        config_quantization: Dict,
) -> List[str]:
    """
    Return the names of quantizable nodes in the model.

    Op types are taken from config_quantization.quantization_args.op_types_to_quantize
    if defined; otherwise all op types supported by ONNX Runtime's quantization
    registries (QDQRegistry + QLinearOpsRegistry) are used as fallback.
    """
    model = onnx.load(model_path)
    quantization_args = config_quantization.get("quantization_args", {})
    configured_ops = set(quantization_args.get("op_types_to_quantize", []))
    allowed_ops = configured_ops if configured_ops else SUPPORTED_QUANT_OP_TYPES

    node_names: List[str] = []
    for node in model.graph.node:
        if not node.name:
            logging.debug("Skipping unnamed node of op_type '%s'", node.op_type)
            continue
        if node.op_type not in allowed_ops:
            continue
        node_names.append(node.name)

    if not node_names:
        raise ValueError(
            f"No quantizable named nodes found in '{model_path}' "
            f"for op types: {allowed_ops}"
        )

    logging.info(
        "Found %d quantizable nodes (%s) in model.",
        len(node_names),
        ", ".join(sorted(allowed_ops)),
    )
    return node_names


def preprocess_model(
        onnx_file: str, quantization_config: DictConfig, oat_work_dir: str
) -> str:
    """Run ``quant_pre_process`` once (if enabled) and return the model path to use."""
    if not quantization_config.get("preprocess", True):
        return onnx_file

    preprocessed_path = os.path.join(oat_work_dir, "preprocessed_model.onnx")
    quant_pre_process(
        onnx_file,
        preprocessed_path,
        **quantization_config.get("preprocess_args", {}),
    )
    logging.info("Preprocessing completed: %s", preprocessed_path)
    return preprocessed_path


def compute_top1_accuracy(
        batch_logits: List[np.ndarray],
        batch_labels: List[np.ndarray],
        num_classes: int,
) -> float:
    """Top-1 accuracy over the cached batches"""
    metric = MulticlassAccuracy(num_classes=num_classes, average="micro")
    for logits, labels in zip(batch_logits, batch_labels):
        metric.update(to_2d_float_tensor(logits), torch.from_numpy(labels))
    return float(metric.compute().item())


def precompute_fp32_baseline(
        session_fp32: ort.InferenceSession,
        output_names: List[str],
        test_dataloader: DataLoader,
        max_test_batches: Optional[int],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Cache test inputs, labels and FP32 outputs so every OAT node sees the same data."""
    logging.info("Precomputing FP32 baseline outputs...")
    batch_inputs: List[np.ndarray] = []
    batch_labels: List[np.ndarray] = []
    batch_outputs_fp32: List[np.ndarray] = []
    for i, batch in enumerate(tqdm(test_dataloader, desc="FP32 baseline")):
        if max_test_batches and i >= max_test_batches:
            break
        images = batch[0].numpy()
        batch_inputs.append(images)
        batch_labels.append(batch[1].numpy())
        batch_outputs_fp32.append(session_fp32.run(output_names, {"input": images})[0])

    if not batch_inputs:
        raise ValueError("No test batches available for OAT sensitivity analysis")
    return batch_inputs, batch_labels, batch_outputs_fp32


def describe_model(model_path: str) -> Dict[str, float]:
    """Size, parameter count, node count and opset of an ONNX model"""
    model = onnx.load(model_path, load_external_data=False)
    model_dir = os.path.dirname(os.path.abspath(model_path))

    total_parameters = 0
    external_files = set()
    for initializer in model.graph.initializer:
        count = 1
        for dim in initializer.dims:
            count *= dim
        total_parameters += count
        if initializer.data_location == onnx.TensorProto.EXTERNAL:
            for entry in initializer.external_data:
                if entry.key == "location":
                    external_files.add(entry.value)

    size_bytes = os.path.getsize(model_path)
    for relative_path in external_files:
        candidate = os.path.join(model_dir, relative_path)
        if os.path.exists(candidate):
            size_bytes += os.path.getsize(candidate)
        else:
            logging.warning("External data file not found: %s", candidate)

    return {
        "model_size_mb": size_bytes / (1024 * 1024),
        "total_parameters": float(total_parameters),
        "num_nodes": float(len(model.graph.node)),
        "opset_version": float(model.opset_import[0].version)
        if model.opset_import
        else float("nan"),
    }


@dataclass(frozen=True)
class EvaluationContext:
    """Everything that is fixed for a model+data pair across quantization scopes"""

    preprocessed_path: str
    session_options: ort.SessionOptions
    output_names: List[str]
    quantization_config: DictConfig
    tensors_range: TensorsData
    batch_inputs: List[np.ndarray]
    batch_labels: List[np.ndarray]
    batch_outputs_fp32: List[np.ndarray]
    num_classes: int
    acc_fp32: float
    latency_fp32: float
    warmup_runs: int
    # For evaluate_on_test_set, which re-reads the test set per model exactly
    # as the paper's evaluation did instead of using the cached batches.
    onnx_file: str = ""
    data_config: Any = None
    max_test_batches: Optional[int] = None


def build_evaluation_context(
        config: DictConfig, work_dir: str, cache_test_set: bool = True
) -> EvaluationContext:
    """Preprocess, calibrate and, optionally, cache the test set with its FP32 reference"""
    quantization_config = config.get("quantization")
    if not quantization_config:
        raise ValueError("Missing 'quantization' in configuration")
    onnx_file = config.get("onnx_file")
    if not onnx_file:
        raise ValueError("Missing 'onnx_file' in configuration")

    session_options = get_session_options(config.get("session_options", {}))
    warmup_runs = int(config.get("warmup_runs", 10))
    max_test_batches = config.get("max_test_batches", None)
    os.makedirs(work_dir, exist_ok=True)
    preprocessed_path = preprocess_model(onnx_file, quantization_config, work_dir)

    session_fp32 = ort.InferenceSession(preprocessed_path, session_options)
    output_names = [output.name for output in session_fp32.get_outputs()]
    if not output_names:
        raise ValueError("ONNX model has no output tensors")

    datamodule: L.LightningDataModule = hydra.utils.instantiate(config.data)
    datamodule.setup("test")
    num_classes = int(datamodule.num_classes)

    batch_inputs: List[np.ndarray] = []
    batch_labels: List[np.ndarray] = []
    batch_outputs_fp32: List[np.ndarray] = []
    acc_fp32 = latency_fp32 = float("nan")
    if cache_test_set:
        batch_inputs, batch_labels, batch_outputs_fp32 = precompute_fp32_baseline(
            session_fp32, output_names, datamodule.test_dataloader(), max_test_batches
        )
        acc_fp32 = compute_top1_accuracy(batch_outputs_fp32, batch_labels, num_classes)
        latency_fp32 = measure_latency(
            preprocessed_path, batch_inputs[0], session_options, warmup_runs=warmup_runs
        )
        logging.info(
            "FP32 baseline: top-1 accuracy %.4f, latency %.6f s over %d batches.",
            acc_fp32,
            latency_fp32,
            len(batch_inputs),
        )
    del session_fp32

    # Once for the whole run: the calibration ranges do not depend on which
    # nodes end up quantized, only on the model and the calibration data.
    logging.info("Calibrating the full model...")
    tensors_range = calibrate_once(preprocessed_path, quantization_config, work_dir)

    return EvaluationContext(
        preprocessed_path=preprocessed_path,
        session_options=session_options,
        output_names=output_names,
        quantization_config=quantization_config,
        tensors_range=tensors_range,
        batch_inputs=batch_inputs,
        batch_labels=batch_labels,
        batch_outputs_fp32=batch_outputs_fp32,
        num_classes=num_classes,
        acc_fp32=acc_fp32,
        latency_fp32=latency_fp32,
        warmup_runs=warmup_runs,
        onnx_file=onnx_file,
        data_config=config.data,
        max_test_batches=max_test_batches,
    )


ORT_TYPE_NP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(double)": np.float64,
}
PAPER_WARMUP_RUNS = 5


def _model_input_shape(model_path: str) -> Tuple[int, ...]:
    model = onnx.load(model_path, load_external_data=False)
    dims = model.graph.input[0].type.tensor_type.shape.dim
    return tuple(dim.dim_value or 1 for dim in dims)


def evaluate_on_test_set(model_path: str, ctx: EvaluationContext) -> Tuple[float, float]:
    """Top-1 accuracy and median latency"""
    session = ort.InferenceSession(
        model_path, ctx.session_options, providers=["CPUExecutionProvider"]
    )
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    output_name = session.get_outputs()[0].name
    target_dtype = ORT_TYPE_NP.get(input_meta.type)

    warmup_input = np.random.randn(1, *_model_input_shape(model_path)[1:]).astype(
        np.float32
    )
    for _ in range(PAPER_WARMUP_RUNS):
        session.run([output_name], {input_name: warmup_input})

    datamodule: L.LightningDataModule = hydra.utils.instantiate(ctx.data_config)
    datamodule.setup("test")
    nc = ctx.num_classes
    # The same MetricCollection as TimmClassification.test_metric, computed on
    # every batch like the paper's progress bar did.
    test_metric = MetricCollection(
        {
            "acc": Accuracy(task="multiclass", num_classes=nc),
            "precision": Precision(task="multiclass", average="macro", num_classes=nc),
            "recall": Recall(task="multiclass", average="macro", num_classes=nc),
            "f1": F1Score(task="multiclass", average="macro", num_classes=nc),
        }
    )

    time_inference: List[float] = []
    targets_inference: List[np.ndarray] = []
    outputs_inference: List[np.ndarray] = []
    pbar = tqdm(datamodule.test_dataloader(), desc="Evaluating", unit="batch")
    for i, batch in enumerate(pbar):
        if ctx.max_test_batches is not None and i >= ctx.max_test_batches:
            break
        images, targets = batch[0], batch[1]
        images_np = images.numpy() if isinstance(images, torch.Tensor) else images
        if not images_np.flags["C_CONTIGUOUS"]:
            images_np = np.ascontiguousarray(images_np)
        if target_dtype is not None and images_np.dtype != target_dtype:
            images_np = images_np.astype(target_dtype)

        ta = time.perf_counter()
        outputs = session.run([output_name], {input_name: images_np})
        tb = time.perf_counter()

        outputs_tensor = torch.tensor(outputs[0])
        if outputs_tensor.dim() > 2:
            outputs_tensor = outputs_tensor.reshape(-1, nc)
        test_metric.update(outputs_tensor, targets)

        acc = test_metric.compute()
        pbar.set_postfix(acc=f"{acc['acc'].item():.4f}")
        time_inference.append(tb - ta)
        targets_inference.append(targets.detach().cpu().numpy())
        outputs_inference.append(outputs_tensor.detach().cpu().numpy())

    del session
    if not time_inference:
        raise ValueError("No test batch was evaluated")
    return float(test_metric.compute()["acc"].item()), float(np.median(time_inference))


RESULT_KEYS = (
    "acc",
    "acc_drop",
    "latency",
    "speedup",
    "n_quantized_layers",
    "model_size_mb",
    "qdq_total_ms",
    "qdq_num_layers",
    "qdq_pct",
)


def empty_result(**overrides: float) -> Dict[str, float]:
    row = {key: float("nan") for key in RESULT_KEYS}
    row.update(overrides)
    return row


def baseline_result(ctx: EvaluationContext) -> Dict[str, float]:
    acc, latency = evaluate_on_test_set(ctx.onnx_file, ctx)
    logging.info("FP32 base: top-1 accuracy %.4f, median latency %.6f s", acc, latency)
    return empty_result(
        acc=acc,
        acc_drop=0.0,
        latency=latency,
        speedup=1.0,
        n_quantized_layers=0.0,
        model_size_mb=describe_model(ctx.onnx_file)["model_size_mb"],
        qdq_total_ms=0.0,
        qdq_num_layers=0.0,
        qdq_pct=0.0,
    )


def _quantized_node_count(ctx: EvaluationContext, node_names: List[str]) -> int:
    if node_names:
        return len(node_names)
    return len(
        get_quantizable_nodes(ctx.onnx_file or ctx.preprocessed_path, ctx.quantization_config)
    )


def evaluate_int8(
        ctx: EvaluationContext,
        node_names: List[str],
        work_dir: str,
        reference: Dict[str, float],
        profile: bool = True,
) -> Dict[str, float]:
    """Quantize ``node_names`` and measure the result against ``reference``"""
    os.makedirs(work_dir, exist_ok=True)
    model_path = os.path.join(work_dir, "quantized_model.onnx")

    try:
        quantize_nodes(
            model_path=ctx.preprocessed_path,
            node_names=node_names,
            tensors_range=ctx.tensors_range,
            config_quantization=ctx.quantization_config,
            output_path=model_path,
        )
    except Exception as exc:
        logging.warning("Error quantizing %s: %s", _scope_label(node_names), exc)
        return empty_result(n_quantized_layers=float(_quantized_node_count(ctx, node_names)))

    acc, latency = evaluate_on_test_set(model_path, ctx)

    overhead = {}
    if profile:
        # Random input of the model's shape
        sample = np.random.randn(*_model_input_shape(model_path)).astype(np.float32)
        trace_path = profile_model(model_path, work_dir, sample, ctx.session_options)
        overhead = measure_qdq_overhead(trace_path)

    return empty_result(
        acc=acc,
        acc_drop=reference["acc"] - acc,
        latency=latency,
        speedup=reference["latency"] / latency if latency > 0 else float("nan"),
        n_quantized_layers=float(_quantized_node_count(ctx, node_names)),
        model_size_mb=describe_model(model_path)["model_size_mb"],
        qdq_total_ms=overhead.get("qdq_total_ms", float("nan")),
        qdq_num_layers=overhead.get("qdq_num_layers", float("nan")),
        qdq_pct=overhead.get("qdq_pct", float("nan")),
    )


def delete_onnx_files(directory: str) -> None:
    """Drop the intermediate ONNX models a sweep leaves behind"""
    removed = 0
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".onnx"):
                os.remove(os.path.join(dirpath, filename))
                removed += 1
    logging.info("Deleted %d intermediate ONNX file(s) under %s", removed, directory)
