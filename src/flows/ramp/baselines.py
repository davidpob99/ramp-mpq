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
from typing import Dict

import wandb

from src.quantization.evaluation import EvaluationContext, evaluate_int8


def run_base(base: Dict[str, float], run: wandb.Run) -> Dict[str, float]:
    """Log the FP32 reference, measured once by the flow on the original ONNX."""
    wandb.run.tags = wandb.run.tags + ("base",)
    row = base
    run.log({f"base/{key}": value for key, value in row.items()})
    logging.info(
        "FP32 base: acc %.4f, latency %.6f s, %.1f MB",
        row["acc"],
        row["latency"],
        row["model_size_mb"],
    )
    return row


def quantize_all(
        ctx: EvaluationContext,
        run: wandb.Run,
        work_dir: str,
        reference: Dict[str, float],
) -> Dict[str, float]:
    """Every quantizable node in INT8.

    Passing no node list lets ONNX Runtime quantize everything that matches
    ``op_types_to_quantize``, which is the scope the config already declares.
    """
    wandb.run.tags = wandb.run.tags + ("quantize-all",)
    row = evaluate_int8(ctx, node_names=[], work_dir=work_dir, reference=reference)
    run.log({f"all_int8/{key}": value for key, value in row.items()})
    logging.info(
        "All INT8: acc %.4f (drop %.4f), speedup %.3fx, %.1f MB, Q/DQ %.1f%% of profiled time",
        row["acc"],
        row["acc_drop"],
        row["speedup"],
        row["model_size_mb"],
        row["qdq_pct"],
    )
    return row
