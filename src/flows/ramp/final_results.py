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
import math
import os
import socket
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

from src.utils.storage import slug
from src.quantization.evaluation import (
    EvaluationContext,
    evaluate_int8,
    get_quantizable_nodes,
)

# One fewer threshold than clusters: the top cluster's max is the global max,
# which selects every layer and is not a useful cut point.
HW_SAFE_K = 4
METRIC_K = 5
_SEED = 42


def portable_path(path: Any) -> Optional[str]:
    """``path`` relative to the repository, so published files carry no local paths.

    Files outside the repository keep only their name.
    """
    if path is None:
        return None
    path = os.path.abspath(str(path))
    root = os.environ.get("PROJECT_ROOT")
    if root and os.path.commonpath([path, os.path.abspath(root)]) == os.path.abspath(root):
        return os.path.relpath(path, root)
    return os.path.basename(path)


def get_k_thresholds(values, k: int = METRIC_K) -> List[float]:
    """Up to ``k`` ascending cut points, from 1-D K-means on ``values``.

    Clusters into ``k + 1`` groups (clamped to the number of distinct values)
    and returns the per-cluster maxima without the global one.
    """
    array = np.asarray(values, dtype=float).reshape(-1, 1)
    array = array[np.isfinite(array).all(axis=1)]
    if array.size == 0:
        return []
    n_unique = len(np.unique(array))
    n_clusters = min(k + 1, n_unique)
    if n_clusters <= 1:
        return [float(np.max(array))]

    labels = KMeans(n_clusters=n_clusters, random_state=_SEED, n_init="auto").fit_predict(array)
    maxima = sorted(float(array[labels == cluster].max()) for cluster in range(n_clusters))
    return maxima[:-1]


def _sensitivity_scores(df: pd.DataFrame, metrics: Dict[str, bool]) -> pd.DataFrame:
    """Normalise every metric so that a HIGHER value always means MORE sensitive"""
    scores = pd.DataFrame(index=df.index)
    for name, higher_is_more_sensitive in metrics.items():
        if name not in df.columns:
            logging.warning("Metric '%s' is not in the table, skipped.", name)
            continue
        scores[name] = df[name] if higher_is_more_sensitive else -df[name]
    if scores.empty:
        raise ValueError("None of the configured metrics is present in the table")
    return scores


def _add_hw_safe_score(
        df: pd.DataFrame,
        scores: pd.DataFrame,
        latency_metric: str,
        latency_measure: str,
) -> Optional[str]:
    """Add a hardware-aware variant: the score, but +inf where quantizing does not pay.

    A layer is worth quantizing only if it actually runs faster in INT8, judged
    by ``<latency_measure>_fp32 / <latency_measure>_int8``.
    """
    if latency_metric not in scores.columns:
        logging.warning(
            "latency_metric '%s' is not among the metrics, skipping the "
            "hardware-aware variant.",
            latency_metric,
        )
        return None
    fp32, int8 = f"{latency_measure}_fp32", f"{latency_measure}_int8"
    if not {fp32, int8}.issubset(df.columns):
        # Configured on purpose, so a silent skip would drop the variant the
        # method is judged on.
        raise KeyError(
            f"The sensitivity table has no '{fp32}'/'{int8}' columns for "
            f"final_results.latency_measure='{latency_measure}'. Tables rebuilt "
            f"from the paper's runs carry 'dur_s'; tables from this repo's "
            f"sensitivity_measurement carry 'latency'."
        )

    speedup = df[fp32] / df[int8]
    thresholds = get_k_thresholds(scores[latency_metric], HW_SAFE_K)
    max_safe = thresholds[-1] if thresholds else float("inf")
    safe = (speedup > 1) & (scores[latency_metric] < max_safe)

    column = f"hw_{latency_metric}"
    scores[column] = float("inf")
    scores.loc[safe, column] = scores.loc[safe, latency_metric]
    logging.info(
        "Hardware-aware pool (%s): %d of %d layers both speed up and stay below "
        "the top sensitivity band.",
        latency_measure,
        int(safe.sum()),
        len(df),
    )
    return column


def _thresholds_for(series: pd.Series) -> List[float]:
    finite = series[np.isfinite(series)]
    return get_k_thresholds(finite, min(METRIC_K, max(len(finite.unique()), 1)))


POLICY_LABELS = {
    "base": "FP32",
    "all_int8": "INT8 unif.",
    "std": "STD",
    "hawqv2_score": "HAWQ-V2",
    "jsd": "RAMP (JSD)",
    "hw_jsd": "JSD + Latency",
}


def select_knee_point(group: pd.DataFrame, reference: Optional[pd.Series] = None) -> pd.Series:
    """The elbow of one policy's accuracy-latency sweep.

    Accuracy and latency are min-max normalised and the point maximising
    ``acc_norm - lat_norm`` wins.
    """
    n_points = len(group) + (reference is not None)
    if n_points < 3:
        return group.sort_values(by=["acc", "latency"], ascending=[False, True]).iloc[0]

    group = group.sort_values(by="acc", ascending=False).reset_index(drop=True)
    acc = group["acc"].to_numpy(dtype=float)
    lat = group["latency"].to_numpy(dtype=float)
    # The reference widens the ranges; it is not a candidate itself.
    scale_acc, scale_lat = acc, lat
    if reference is not None:
        scale_acc = np.append(acc, float(reference["acc"]))
        scale_lat = np.append(lat, float(reference["latency"]))
    acc_min, acc_max = scale_acc.min(), scale_acc.max()
    lat_min, lat_max = scale_lat.min(), scale_lat.max()
    if acc_max - acc_min < 1e-6 or lat_max - lat_min < 1e-6:
        return group.iloc[0]

    score = (acc - acc_min) / (acc_max - acc_min) - (lat - lat_min) / (lat_max - lat_min)
    return group.iloc[int(np.argmax(score))]


def select_candidates(df_results: pd.DataFrame) -> pd.DataFrame:
    """One candidate per policy using elbow rule"""
    measured = df_results.dropna(subset=["acc", "latency"])
    base = measured[measured["name"] == "base"]
    reference = base.iloc[0] if not base.empty else None
    candidates = []
    for policy in ["base", "all_int8"]:
        rows = measured[measured["name"] == policy]
        if not rows.empty:
            candidates.append((policy, rows.iloc[0]))
    for metric in dict.fromkeys(measured["metric"]):
        if metric == "base":
            continue
        sweep = measured[(measured["metric"] == metric) & (measured["name"] != "all_int8")]
        if not sweep.empty:
            candidates.append((metric, select_knee_point(sweep, reference)))

    table = pd.DataFrame(
        {
            "policy": [POLICY_LABELS.get(key, key) for key, _ in candidates],
            "name": [row["name"] for _, row in candidates],
            "acc_pct": [row["acc"] * 100 for _, row in candidates],
            "latency_ms": [row["latency"] * 1000 for _, row in candidates],
            "speedup": [row["speedup"] for _, row in candidates],
            "acc_drop_pct": [row["acc_drop"] * 100 for _, row in candidates],
            "n_quantized_layers": [row["n_quantized_layers"] for _, row in candidates],
            "model_size_mb": [row["model_size_mb"] for _, row in candidates],
        }
    )
    return table


def _json_value(value: Any) -> Any:
    """NaN is not valid JSON; unmeasured values are written as null."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return _json_value(value.item())
    return value


def _policy(
        row: Dict[str, Any],
        quantized: Iterable[str],
        all_nodes: List[str],
) -> Dict[str, Any]:
    """One configuration as a per-layer precision map, plus what it measured.

    Covers every quantizable node of the model, not only the layers of the
    sensitivity table: a deployment needs the precision of each of them.
    """
    int8 = set(quantized)
    # Defensive: a selected node the op-type scan did not list is still INT8.
    nodes = list(all_nodes) + sorted(int8 - set(all_nodes))
    layers = {node: ("int8" if node in int8 else "fp32") for node in nodes}
    return {
        "name": row["name"],
        "metric": row["metric"],
        "threshold": _json_value(row["threshold"]),
        "n_int8": sum(p == "int8" for p in layers.values()),
        "n_fp32": sum(p == "fp32" for p in layers.values()),
        "results": {
            key: _json_value(value)
            for key, value in row.items()
            if key not in ("name", "metric", "threshold", "layers")
        },
        "layers": layers,
    }


def final_results(
        config: DictConfig,
        ctx: EvaluationContext,
        output_dir: str,
        run: wandb.Run,
        reference: Dict[str, float],
) -> pd.DataFrame:
    """Sweep quantize-set sizes for every configured metric and log the results."""
    wandb.run.tags = wandb.run.tags + ("final-results",)
    final_config = config.get("final_results", {})

    table_path = config.get("sensitivity_table")
    df = pd.read_csv(table_path)
    layer_column = "node" if "node" in df.columns else "layer"
    if layer_column not in df.columns:
        raise KeyError(
            f"'{table_path}' has no 'node' or 'layer' column to identify layers"
        )
    logging.info("Sensitivity table: %d layers from %s", len(df), table_path)

    metrics = {
        str(name): bool(value)
        for name, value in OmegaConf.to_container(
            final_config.get("metrics", {}), resolve=True
        ).items()
    }
    if not metrics:
        raise ValueError("Missing 'final_results.metrics' in configuration")

    scores = _sensitivity_scores(df, metrics)
    latency_measure = str(final_config.get("latency_measure", "dur_s"))
    hw_column = _add_hw_safe_score(
        df, scores, str(final_config.get("latency_metric", "")), latency_measure
    )
    sweep_columns = [c for c in scores.columns if c != hw_column]
    if hw_column:
        sweep_columns.append(hw_column)

    work_root = os.path.join(output_dir, "final_results")
    rows: List[Dict] = []
    # Every node the configured op types allow: all_int8 quantizes exactly these.
    all_nodes = get_quantizable_nodes(config.get("onnx_file"), ctx.quantization_config)
    policies: List[Dict[str, Any]] = []

    # The FP32 reference: no layer quantized at all, measured once by the flow.
    base_row = reference
    rows.append(
        {"name": "base", "metric": "base", "threshold": float("nan"), "layers": [], **base_row}
    )
    policies.append(_policy(rows[-1], [], all_nodes))

    all_int8: Optional[Dict[str, float]] = None
    for metric in sweep_columns:
        # The first threshold of every metric is -1: it selects no layer
        # explicitly, which falls back to the config's own quantization scope,
        # i.e. the fully quantized model. Evaluated once, reused across metrics.
        thresholds = [-1.0] + _thresholds_for(scores[metric])
        logging.info("%s: %d thresholds", metric, len(thresholds))
        for index, threshold in enumerate(tqdm(thresholds, desc=f"sweep {metric}")):
            if index == 0:
                # Normalised scores can be negative (negated SNR-like metrics),
                # so the empty selection is set explicitly, not via score < -1.
                selected: List[str] = []
                name = "all_int8"
                if all_int8 is None:
                    all_int8 = evaluate_int8(
                        ctx, selected, os.path.join(work_root, name), reference
                    )
                row = all_int8
            else:
                selected = df.loc[scores[metric] < threshold, layer_column].tolist()
                name = f"{metric}_{index}"
                if not selected:
                    # An empty list means "the whole model" to evaluate_int8, so
                    # this would silently re-run all_int8 under another name.
                    # No layer quantized is the base row, already in the table.
                    logging.info("%s selects no layer, skipped.", name)
                    continue
                row = evaluate_int8(
                    ctx, selected, os.path.join(work_root, name), reference
                )
            rows.append(
                {
                    "name": name,
                    "metric": metric,
                    "threshold": float(threshold),
                    "layers": selected,
                    **row,
                }
            )
            # all_int8 passes an empty list, which means "every quantizable node".
            policies.append(
                _policy(rows[-1], all_nodes if name == "all_int8" else selected, all_nodes)
            )

    df_results = pd.DataFrame(rows)
    results_path = os.path.join(output_dir, "final_results.csv")
    df_results.to_csv(results_path, index=False)

    selected = select_candidates(df_results)
    selected_path = os.path.join(output_dir, "final_results_selected.csv")
    selected.to_csv(selected_path, index=False)
    logging.info("Selected candidates (elbow per policy):\n%s", selected.to_string(index=False))

    # Mark the chosen configurations in the policy file. all_int8 appears once
    # per sweep there; only its first entry is marked.
    chosen = dict(zip(selected["name"], selected["policy"]))
    for policy in policies:
        policy["selected_for"] = chosen.pop(policy["name"], None)

    policies_path = os.path.join(output_dir, "quantization_policies.json")
    with open(policies_path, "w") as handle:
        json.dump(
            {
                "model_name": config.get("model", {}).get("model_name"),
                "onnx_file": portable_path(config.get("onnx_file")),
                "sensitivity_table": portable_path(table_path),
                "latency_measure": latency_measure,
                "quantizable_nodes": len(all_nodes),
                "policies": policies,
            },
            handle,
            indent=2,
        )
    logging.info("Quantization policies: %d written to %s", len(policies), policies_path)

    run.log(
        {
            "final_results/table": wandb.Table(dataframe=df_results.astype({"layers": str})),
            "final_results/selected": wandb.Table(dataframe=selected),
        }
    )
    # One collection per model, versioned across runs; the host alias points
    # at the latest version from each device (final-results-<model>:<host>).
    hostname = socket.gethostname()
    artifact = wandb.Artifact(
        name=f"final-results-{slug(config.model.model_name)}",
        type="analysis",
        metadata={
            "sensitivity_table": portable_path(table_path),
            "num_layers": str(len(df)),
            "metrics": str(sweep_columns),
            "host": hostname,
        },
    )
    artifact.add_file(results_path)
    artifact.add_file(selected_path)
    artifact.add_file(policies_path)
    run.log_artifact(artifact, aliases=["latest", slug(hostname)])

    logging.info("Final results: %d configurations evaluated.", len(df_results))
    return df_results
