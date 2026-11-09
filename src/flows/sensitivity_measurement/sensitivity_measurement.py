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

import socket

import hydra
import rootutils
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import wandb

# Puts the repo root on sys.path so `src` imports whether this file is run as a
# module, by path, or from an IDE with a different working directory.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.flows.sensitivity_measurement.oat import analyze_oat  # noqa: E402
from src.flows.sensitivity_measurement.profile_layers import (  # noqa: E402
    analyze_layer_profiling,
)
from src.flows.sensitivity_measurement.recall_top_k import (  # noqa: E402
    analyze_recall_top_k,
)
from src.flows.sensitivity_measurement.weights import analyze_weights  # noqa: E402
from src.flows.sensitivity_measurement.weights_pytorch import (  # noqa: E402
    analyze_weights_pytorch,
)
from src.utils.tracking import wandb_run_config  # noqa: E402

def _get_run_name(config: DictConfig) -> str:
    """Build an experiment run name as ``<model_name>-<hostname>``.

    How much test data a run saw is in ``max_test_batches``, which W&B records
    with the rest of the config, so it does not need repeating here.
    """
    hostname = socket.gethostname()
    model_name = config.get("model").get("model_name")
    return f"{model_name}-{hostname}"

@hydra.main(
    version_base="1.3",
    config_path="../../../config",
    config_name="sensitivity_measurement.yaml",
)
def sensitivity_measurement(config: DictConfig) -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")
    output_dir = HydraConfig.get().runtime.output_dir
    wandb_config = config.logger.get("wandb", {})
    run = wandb.init(
        project=wandb_config.get("project", "sensitivity_measurement"),
        entity=wandb_config.get("entity", None),
        name=_get_run_name(config),
        config=wandb_run_config(config),
        dir=output_dir,
        mode=wandb_config.get("mode", None),
    )
    if run is None:
        raise RuntimeError("Failed to initialize W&B run")

    steps_config = config.get("steps", config.get("step", {}))

    if steps_config.get("analyze_weights", False):
        analyze_weights(config, output_dir, run)
    if steps_config.get("analyze_weights_pytorch", False):
        analyze_weights_pytorch(config, output_dir, run)
    if steps_config.get("analyze_oat", False):
        analyze_oat(config, output_dir, run)
    if steps_config.get("profile_layers", False):
        analyze_layer_profiling(config, output_dir, run)
    # Last: it consumes the CSVs the steps above write.
    if steps_config.get("recall_top_k", False):
        analyze_recall_top_k(config, output_dir, run)

    run.finish()


if __name__ == "__main__":
    sensitivity_measurement()
