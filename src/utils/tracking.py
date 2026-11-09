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
from typing import Any, Dict, List

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def _to_container_lenient(node: Any, path: str, unresolved: List[str]) -> Any:
    """Resolve what can be resolved; keep the raw text of what cannot.

    The model configs carry training-only interpolations such as
    ``${trainer.max_epochs}`` that no flow here defines, so a strict resolve
    fails. Tracking must never take a run down, so those values are recorded
    as their raw ``${...}`` string instead.
    """
    if OmegaConf.is_dict(node):
        items = [(key, f"{path}.{key}" if path else str(key)) for key in node.keys()]
    elif OmegaConf.is_list(node):
        items = [(index, f"{path}[{index}]") for index in range(len(node))]
    else:
        return node

    raw = OmegaConf.to_container(node, resolve=False)
    resolved = {} if OmegaConf.is_dict(node) else []
    for key, child_path in items:
        try:
            value = node[key]
        except Exception:
            unresolved.append(child_path)
            value = raw[key]
        else:
            if OmegaConf.is_config(value):
                value = _to_container_lenient(value, child_path, unresolved)
        if isinstance(resolved, dict):
            resolved[key] = value
        else:
            resolved.append(value)
    return resolved


def wandb_run_config(config: DictConfig) -> Dict[str, Any]:
    """The resolved Hydra config plus the CLI overrides, for ``wandb.init``.

    Without it a run only records its name, which is built from
    ``model.model_name`` -- nothing shows which ONNX file, test subset or node set
    it actually used, so a run named after one model can silently analyse
    another. The overrides record how the run was launched.
    """
    unresolved: List[str] = []
    resolved = _to_container_lenient(config, "", unresolved)
    if unresolved:
        logging.info(
            "Logged to W&B unresolved (training-only or missing): %s",
            ", ".join(unresolved),
        )
    resolved["hydra_overrides"] = list(HydraConfig.get().overrides.task)
    return resolved
