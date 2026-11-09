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
from typing import Dict, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

import wandb
from src.metrics.ranking import compute_recall_at_k
from src.quantization.profiling import profiled_node_intersection
from src.utils.onnx import onnx_to_pytorch_name

WEIGHTS_CSV = "weights_per_node_stats.csv"
HAWQ_CSV = "hawqv2_scores.csv"
OAT_CSV = "oat_sensitivity_results.csv"
PROFILING_CSV = "layer_profiling.csv"


def _read_if_exists(output_dir: str, filename: str) -> Optional[pd.DataFrame]:
    """Load a step's CSV, or None when that step did not run."""
    path = os.path.join(output_dir, filename)
    if not os.path.exists(path):
        logging.info("'%s' not found — that step did not run.", filename)
        return None
    return pd.read_csv(path)


def _profiled_nodes(profiling_filter) -> Optional[set]:
    """Node names the profiling filter keeps, or None when it is not configured.

    Either a CSV with a ``node`` column (the precomputed intersection of an
    FP32 and an INT8 trace, which is what ships with the repo because the raw
    traces run to tens of MB) or a mapping with ``fp32`` and ``int8`` paths to
    rebuild it from traces.
    """
    if not profiling_filter:
        return None
    if isinstance(profiling_filter, str):
        nodes = set(pd.read_csv(profiling_filter)["node"])
        logging.info("Profiling filter: %d nodes from %s", len(nodes), profiling_filter)
        return nodes
    fp32 = profiling_filter.get("fp32")
    int8 = profiling_filter.get("int8")
    if not fp32 or not int8:
        raise ValueError(
            "recall_top_k.profiling_filter needs a CSV path or both 'fp32' and "
            "'int8' trace paths"
        )
    return profiled_node_intersection(fp32, int8)


def _build_layer_table(
        output_dir: str, profiling_filter=None
) -> Optional[pd.DataFrame]:
    """Join the per-layer outputs of the other steps into a single table.

    A layer is ranked only if it is present in every table, so all predictors compete on the same
    population and share the same K.
    """
    df_oat = _read_if_exists(output_dir, OAT_CSV)
    if df_oat is None:
        logging.warning(
            "Skipping recall: '%s' is required for the ground truth. "
            "Enable steps.analyze_oat.",
            OAT_CSV,
        )
        return None

    df = df_oat.rename(columns={"layer": "node"})
    logging.info("Layer population: %d OAT nodes", len(df))

    # Optional, and hardware-dependent: keeps only the nodes that ORT did not
    # fuse away, on the machine the traces came from.
    profiled = _profiled_nodes(profiling_filter)
    if profiled is not None:
        df = df[df["node"].isin(profiled)]
        logging.info("Layer population: %d after the profiling filter", len(df))
        if df.empty:
            raise ValueError(
                "No layer survived the profiling filter: the traces do not "
                "match this model."
            )

    # Weights stats share the ONNX node namespace, so this join is exact.
    df_weights = _read_if_exists(output_dir, WEIGHTS_CSV)
    if df_weights is not None:
        df = df.merge(df_weights, on="node", how="inner", suffixes=("", "_weights"))
        logging.info("Layer population: %d after joining weight stats", len(df))

    # HAWQ-V2 keys on PyTorch modules and needs the name bridge.
    df_hawq = _read_if_exists(output_dir, HAWQ_CSV)
    if df_hawq is not None:
        df_hawq = df_hawq.assign(
            pytorch_layer=df_hawq["pytorch_layer"].str.replace(
                "model.", "", regex=False
            )
        )
        df["pytorch_layer"] = df["node"].apply(onnx_to_pytorch_name)
        df = df.merge(df_hawq, on="pytorch_layer", how="inner")
        logging.info(
            "Layer population: %d after joining HAWQ-V2 (%d HAWQ-V2 layers)",
            len(df),
            len(df_hawq),
        )
        if df.empty:
            raise ValueError(
                "No layer survived the HAWQ-V2 join: the ONNX->PyTorch name "
                "pattern does not fit this export."
            )

    # Per-node kernel times of this device. A left join: it adds columns
    # without changing which layers are ranked. A node missing from the traces
    # (fused away) keeps an empty dur_s and never enters ramp's hardware pool.
    df_profiling = _read_if_exists(output_dir, PROFILING_CSV)
    if df_profiling is not None:
        df = df.merge(df_profiling, on="node", how="left")
        logging.info(
            "Layer profiling joined: %d of %d layers have no dur_s",
            int(df["dur_s_fp32"].isna().sum()),
            len(df),
        )
    return df


def analyze_recall_top_k(config: DictConfig, output_dir: str, run: wandb.Run) -> None:
    """Rank each proxy metric by how much of the true top-K sensitive layers it recovers"""
    wandb.run.tags = wandb.run.tags + ("recall-top-k",)
    recall_config = config.get("recall_top_k", {})
    predictors: Dict[str, bool] = {
        str(name): bool(higher_is_more_sensitive)
        for name, higher_is_more_sensitive in OmegaConf.to_container(
            recall_config.get("predictors", {}), resolve=True
        ).items()
    }
    if not predictors:
        raise ValueError("Missing 'recall_top_k.predictors' in configuration")

    fractions = tuple(
        float(f) for f in recall_config.get("fractions", [0.10, 0.20, 0.30])
    )
    ground_truth = str(recall_config.get("ground_truth", "acc_drop"))

    df_layers = _build_layer_table(
        output_dir, recall_config.get("profiling_filter", None)
    )
    if df_layers is None:
        return

    df_recall = compute_recall_at_k(
        df=df_layers,
        predictors=predictors,
        ground_truth=ground_truth,
        fractions=fractions,
    )

    layers_csv_path = os.path.join(output_dir, "sensitivity_layer_table.csv")
    df_layers.to_csv(layers_csv_path, index=False)
    recall_csv_path = os.path.join(output_dir, "recall_top_k.csv")
    df_recall.to_csv(recall_csv_path)

    run.log(
        {
            "recall/table": wandb.Table(dataframe=df_recall.reset_index()),
            "recall/layer_table": wandb.Table(dataframe=df_layers),
        }
    )

    artifact = wandb.Artifact(
        name="recall-top-k",
        type="analysis",
        metadata={
            "ground_truth": ground_truth,
            "fractions": str(list(fractions)),
            "num_layers": str(len(df_layers)),
            "num_predictors": str(len(df_recall)),
        },
    )
    artifact.add_file(recall_csv_path)
    artifact.add_file(layers_csv_path)
    run.log_artifact(artifact)

    logging.info("Recall @ top-K computed for %d predictors.", len(df_recall))
