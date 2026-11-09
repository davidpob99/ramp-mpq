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
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import pandas as pd
from omegaconf import DictConfig

import wandb
from src.utils.onnx import build_node_weight_arrays, build_node_weight_map
from src.utils.pytorch import prepare_weight_for_channel_metrics
from src.metrics.signal import compute_estimated_sqnr, compute_channel_range_cv, compute_range_ratio


def _compute_onnx_weights_std(model: onnx.ModelProto) -> Dict[str, float]:
    """Compute the global std of all numeric ONNX weights (grouped by node)"""
    logging.info("Analyzing weights...")
    node_weight_map = build_node_weight_map(model)

    total_values = 0
    total_sum = 0.0
    total_sum_sq = 0.0
    for values_f64 in node_weight_map.values():
        total_values += int(values_f64.size)
        total_sum += float(values_f64.sum())
        total_sum_sq += float(np.square(values_f64).sum())

    if total_values == 0:
        raise ValueError("No numeric ONNX node weights found to compute std")

    mean = total_sum / total_values
    variance = max(total_sum_sq / total_values - mean * mean, 0.0)

    return {
        "std": float(np.sqrt(variance)),
        "values": float(total_values),
        "nodes": float(len(node_weight_map)),
    }


def _sanitize_wandb_key(raw_name: str) -> str:
    """Sanitize dynamic metric names for W&B logging keys"""
    sanitized = []
    for char in raw_name:
        if char.isalnum() or char in {"_", "-"}:
            sanitized.append(char)
        else:
            sanitized.append("_")
    return "".join(sanitized)


def _select_primary_weight(weight_arrays: List[np.ndarray]) -> Optional[np.ndarray]:
    """Pick a representative weight array for per-channel statistics"""
    if not weight_arrays:
        return None
    candidates = [weights for weights in weight_arrays if weights.ndim >= 2]
    pool = candidates if candidates else weight_arrays
    return max(pool, key=lambda item: item.size)


def _collect_node_weight_stats(
    model: onnx.ModelProto,
    max_samples_per_node: int,
    random_seed: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Collect per-node weight statistics and sampled values for plotting"""
    node_weight_map = build_node_weight_map(model)
    node_weight_arrays = build_node_weight_arrays(model)
    rng = np.random.default_rng(random_seed)
    node_rows: List[Dict[str, float | str]] = []
    node_samples: Dict[str, np.ndarray] = {}

    for topo_idx, (node_name, flattened) in enumerate(node_weight_map.items()):
        sample_size = min(flattened.size, max_samples_per_node)
        if sample_size == flattened.size:
            sampled = flattened.astype(np.float32, copy=False)
        else:
            sample_idx = rng.choice(flattened.size, size=sample_size, replace=False)
            sampled = flattened[sample_idx].astype(np.float32, copy=False)

        primary_weight = _select_primary_weight(node_weight_arrays.get(node_name, []))
        if primary_weight is not None:
            weight_tensor = prepare_weight_for_channel_metrics(primary_weight)
            range_ratio = compute_range_ratio(weight_tensor)
            channel_range_cv = compute_channel_range_cv(weight_tensor)
            estimated_sqnr = compute_estimated_sqnr(weight_tensor)
        else:
            range_ratio = float("nan")
            channel_range_cv = float("nan")
            estimated_sqnr = float("nan")

        node_rows.append(
            {
                "node": node_name,
                "topo_idx": float(topo_idx),
                "std": float(np.std(primary_weight))
                if primary_weight is not None
                else float("nan"),
                "mean": float(np.mean(primary_weight))
                if primary_weight is not None
                else float("nan"),
                "min": float(np.min(primary_weight))
                if primary_weight is not None
                else float("nan"),
                "max": float(np.max(primary_weight))
                if primary_weight is not None
                else float("nan"),
                "value_count": float(flattened.size),
                "sample_count": float(sampled.size),
                "range_ratio": range_ratio,
                "channel_range_cv": channel_range_cv,
                "estimated_sqnr": estimated_sqnr,
            }
        )
        node_samples[node_name] = sampled

    if not node_rows:
        raise ValueError("No numeric ONNX node weights found for per-node analysis")

    df_node_stats = pd.DataFrame(node_rows).sort_values(by="topo_idx", ascending=True)
    return df_node_stats.reset_index(drop=True), node_samples


def analyze_weights(config: DictConfig, output_dir: str, run: wandb.Run) -> None:
    wandb.run.tags = wandb.run.tags + ("analyze-weights",)
    model_path = config.get("onnx_file")
    if not model_path:
        raise ValueError("Missing 'onnx_file' in configuration")

    weights_analysis_config = config.get("analyze_weights", {})
    max_samples_per_node = int(weights_analysis_config.get("max_samples_per_layer", 2048))
    random_seed = int(config.get("seed", 42))
    if max_samples_per_node <= 0:
        raise ValueError("max_samples_per_layer must be greater than 0")

    model = onnx.load(model_path)
    weights_stats = _compute_onnx_weights_std(model)
    df_node_stats, node_samples = _collect_node_weight_stats(
        model=model,
        max_samples_per_node=max_samples_per_node,
        random_seed=random_seed,
    )

    node_names = df_node_stats["node"].tolist()
    node_keys = [f"node_{idx:04d}" for idx, _ in enumerate(node_names)]
    npz_payload: Dict[str, np.ndarray] = {
        "granularity": np.array(["node"], dtype=object),
        "node_names": np.array(node_names, dtype=object),
        "node_keys": np.array(node_keys, dtype=object),
        "std_per_node": df_node_stats["std"].to_numpy(dtype=np.float32),
        "mean_per_node": df_node_stats["mean"].to_numpy(dtype=np.float32),
        "min_per_node": df_node_stats["min"].to_numpy(dtype=np.float32),
        "max_per_node": df_node_stats["max"].to_numpy(dtype=np.float32),
        "value_count_per_node": df_node_stats["value_count"].to_numpy(dtype=np.float32),
        "sample_count_per_node": df_node_stats["sample_count"].to_numpy(
            dtype=np.float32
        ),
    }
    for node_key, node_name in zip(node_keys, node_names):
        npz_payload[node_key] = node_samples[node_name]

    npz_path = os.path.join(output_dir, "weights_per_node_samples.npz")
    np.savez_compressed(npz_path, **npz_payload)

    stats_csv_path = os.path.join(output_dir, "weights_per_node_stats.csv")
    df_node_stats.to_csv(stats_csv_path, index=False)

    run.log(
        {
            "weights/std": weights_stats["std"],
            "weights/value_count": weights_stats["values"],
            "weights/node_count": weights_stats["nodes"],
            "weights/per_node_stats": wandb.Table(dataframe=df_node_stats),
        }
    )

    artifact = wandb.Artifact(
        name="analyze_weights",
        type="analysis",
        metadata={
            "model_path": model_path,
            "granularity": "node",
            "max_samples_per_layer": str(max_samples_per_node),
            "random_seed": str(random_seed),
        },
    )
    artifact.add_file(npz_path)
    artifact.add_file(stats_csv_path)
    run.log_artifact(artifact)

    logging.info("ONNX weights analysis complete")
