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

import re
from typing import Dict, List, Optional

import numpy as np
import onnx


def build_node_weight_arrays(
        model: onnx.ModelProto,
) -> Dict[str, List[np.ndarray]]:
    """Map each ONNX node name to its list of numeric weight (initializer) arrays.

    Non-numeric or empty initializers are skipped. Unnamed nodes get a
    ``<op_type>_<index>`` fallback name. Nodes without weights are omitted.
    """
    initializer_map: Dict[str, np.ndarray] = {}
    for initializer in model.graph.initializer:
        values = onnx.numpy_helper.to_array(initializer)
        if values.size == 0 or not np.issubdtype(values.dtype, np.number):
            continue
        initializer_map[initializer.name] = values

    node_weight_map: Dict[str, List[np.ndarray]] = {}
    for node_idx, node in enumerate(model.graph.node):
        node_name = node.name or f"{node.op_type}_{node_idx}"
        node_weights = [
            initializer_map[input_name]
            for input_name in node.input
            if input_name in initializer_map
        ]
        if node_weights:
            node_weight_map[node_name] = node_weights

    return node_weight_map


def build_node_weight_map(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    """Map each ONNX node name to its concatenated numeric weight values (float64, 1-D)."""
    node_weight_arrays = build_node_weight_arrays(model)
    node_weight_map: Dict[str, np.ndarray] = {}
    for node_name, node_weights in node_weight_arrays.items():
        if len(node_weights) == 1:
            node_weight_map[node_name] = (
                node_weights[0].astype(np.float64, copy=False).ravel()
            )
            continue
        node_weight_map[node_name] = np.concatenate(
            [weights.astype(np.float64, copy=False).ravel() for weights in node_weights]
        )

    return node_weight_map


def onnx_to_pytorch_name(onnx_name: str) -> Optional[str]:
    """Derive the PyTorch module path from an ONNX node name.

    Bridges the two naming spaces the analysis steps produce: OAT and the weight
    stats key on ONNX nodes (``/model/features/0/Conv``) while the Hessian and
    HAWQ-V2 scores key on PyTorch modules (``features.0``).

    The pattern depends on how the model was exported, so this returns None
    instead of raising when a name does not match; callers should treat the
    match rate as a signal, not assume full coverage.
    """
    if not isinstance(onnx_name, str):
        return None

    match = re.match(r"^/model/(.*)/[A-Za-z0-9]+$", onnx_name)
    if not match:
        return None

    parts = match.group(1).split("/")
    # Drop a scope that is only the prefix of the next one ("features",
    # "features.0" -> keep "features.0"), which ONNX emits for nested modules.
    cleaned = [
        part
        for i, part in enumerate(parts)
        if not (i < len(parts) - 1 and parts[i + 1].startswith(part + "."))
    ]
    return ".".join(cleaned)
