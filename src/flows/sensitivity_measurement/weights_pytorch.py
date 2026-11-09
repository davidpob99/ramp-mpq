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

import hydra
import pandas as pd
import timm
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import wandb
from src.metrics.hawq import compute_hessian_trace_hawq
from src.quantization.pytorch import get_l2_quantization_error


def _resolve_device(config: DictConfig) -> str:
    """Resolve the torch device from ``use_gpu``, falling back to CPU."""
    if not bool(config.get("use_gpu", False)):
        return "cpu"
    if not torch.cuda.is_available():
        logging.warning("'use_gpu' is set but CUDA is unavailable — falling back to CPU")
        return "cpu"
    return "cuda"


def _build_calibration_pytorch(calibration_data, transforms_data, batch_size=1):
    transform = hydra.utils.instantiate(transforms_data)
    dataset = ImageFolder(root=calibration_data.get("folder_path"), transform=transform)

    max_images = calibration_data.get("max_images", len(dataset))
    indices = list(range(min(max_images, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices)

    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return dataloader


def analyze_weights_pytorch(config: DictConfig, output_dir: str, run: wandb.Run) -> None:
    wandb.run.tags = wandb.run.tags + ("analyze-weights-pytorch",)
    pytorch_model = config.get("pytorch_file")
    calibration_dataloader = _build_calibration_pytorch(
        config.get("quantization").get("calibration_data"),
        config.get("data").get("transform_test"),
        config.get("data").get("batch"),
    )

    if pytorch_model is not None and calibration_dataloader is not None:
        logging.info("PyTorch model and dataloader found.")
        device = _resolve_device(config)
        state_dict = torch.load(pytorch_model, weights_only=False, map_location=device)
        if isinstance(state_dict, dict):
            dataset = hydra.utils.instantiate(config.data)
            model = timm.create_model(
                config.model.model_name,
                num_classes=dataset.num_classes,
                pretrained=False,
            )
            model.load_state_dict(state_dict)
        else:
            model = state_dict

        if config.get("hessian_trace", {}).get("enabled", False):
            logging.info("Computing Hessian Trace...")
            criterion = config.get("criterion", torch.nn.CrossEntropyLoss())

            trace_dict = compute_hessian_trace_hawq(
                model=model,
                dataloader=calibration_dataloader,
                criterion=criterion,
                device=device,
                seed=int(config.get("seed", 42)),
            )
            hawq_v2_scores = {}
            for name, module in model.named_modules():
                if name in trace_dict:
                    l2_perturbation = get_l2_quantization_error(module.weight, 8)
                    # Final HAWQ-V2 Omega score
                    hawq_v2_scores[name] = trace_dict[name] * l2_perturbation
                else:
                    logging.warning(f"[Hessian] Layer {name} not found in trace")

            # Convert to DataFrame and save
            df_hessian_stats = pd.DataFrame(
                trace_dict.items(), columns=["pytorch_layer", "hessian_trace"]
            )
            df_hawq_stats = pd.DataFrame(
                hawq_v2_scores.items(), columns=["pytorch_layer", "hawqv2_score"]
            )

            hessian_csv_path = os.path.join(output_dir, "hessian_trace_stats.csv")
            df_hessian_stats.to_csv(hessian_csv_path, index=False)
            run.log({"hessian_trace_stats": wandb.Table(dataframe=df_hessian_stats)})

            hawq_csv_path = os.path.join(output_dir, "hawqv2_scores.csv")
            df_hawq_stats.to_csv(hawq_csv_path, index=False)
            run.log({"hawqv2_scores": wandb.Table(dataframe=df_hawq_stats)})
            logging.info("Hessian trace computed successfully.")
    else:
        logging.info(
            "Skipping PyTorch weight analysis (Hessian trace): provide "
            "'pytorch_file' and calibration data in the config to enable it."
        )
