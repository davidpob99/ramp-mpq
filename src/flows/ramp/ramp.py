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

import hydra
import rootutils
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# Puts the repo root on sys.path so `src` imports whether this file is run as a
# module, by path, or from an IDE with a different working directory.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.flows.ramp.baselines import quantize_all, run_base  # noqa: E402
from src.flows.ramp.final_results import final_results  # noqa: E402
from src.quantization.evaluation import (  # noqa: E402
    baseline_result,
    build_evaluation_context,
    delete_onnx_files,
)
from src.utils.storage import upload_quantized_models  # noqa: E402
from src.utils.tracking import wandb_run_config  # noqa: E402


def _get_run_name(config: DictConfig) -> str:
    """Build an experiment run name as ``<model_name>-<hostname>``."""
    return f"{config.get('model').get('model_name')}-{socket.gethostname()}"


@hydra.main(
    version_base="1.3",
    config_path="../../../config",
    config_name="ramp.yaml",
)
def ramp(config: DictConfig) -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")
    output_dir = HydraConfig.get().runtime.output_dir
    wandb_config = config.logger.get("wandb", {})
    run = wandb.init(
        project=wandb_config.get("project", "ramp"),
        entity=wandb_config.get("entity", None),
        name=_get_run_name(config),
        config=wandb_run_config(config),
        dir=output_dir,
        mode=wandb_config.get("mode", None),
    )
    if run is None:
        raise RuntimeError("Failed to initialize W&B run")

    steps_config = config.get("steps", {})

    # Built once: the preprocessing and the calibration ranges are the same for
    # every configuration the steps below evaluate. No cached test set: each
    # model is evaluated by re-reading it, as the paper did.
    evaluation = build_evaluation_context(
        config, os.path.join(output_dir, "ramp"), cache_test_set=False
    )
    # The FP32 reference every configuration is compared against. A full pass
    # over the test set, so measured once and shared.
    base = baseline_result(evaluation)

    if steps_config.get("base", False):
        run_base(base, run)
    if steps_config.get("quantize_all", False):
        quantize_all(evaluation, run, os.path.join(output_dir, "ramp", "all_int8"), base)
    if steps_config.get("final_results", False):
        final_results(config, evaluation, output_dir, run, base)

    # Before the clean-up, which deletes the local copies. A failed upload
    # must not fail a run that took hours: the models stay on disk and can
    # be uploaded later with upload_quantized_models.
    upload_config = config.get("upload", {})
    if upload_config.get("s3_uri"):
        try:
            upload_quantized_models(
                output_dir,
                run,
                s3_uri=upload_config.s3_uri,
                endpoint_url=upload_config.get("endpoint_url"),
                model_name=config.model.model_name,
                hostname=socket.gethostname(),
            )
        except Exception:
            logging.exception("S3 upload failed; keeping the local ONNX files.")
            config.clean_up.delete_onnx_files = False

    if config.get("clean_up", {}).get("delete_onnx_files", False):
        delete_onnx_files(os.path.join(output_dir, "ramp"))
        delete_onnx_files(os.path.join(output_dir, "final_results"))

    run.finish()


if __name__ == "__main__":
    ramp()
