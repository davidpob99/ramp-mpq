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

import contextlib
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


@contextlib.contextmanager
def _conditional_math_sdp(needs_fallback: bool):
    if needs_fallback:
        # Usamos SDPBackend.MATH para forzar el backend que soporta 2da derivada
        with sdpa_kernel(SDPBackend.MATH):
            yield
    else:
        yield

def _check_needs_sdp_fallback(model, criterion, inputs, targets, weights) -> bool:
    try:
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        grads = torch.autograd.grad(loss, weights, create_graph=True)

        vT_g = sum(torch.sum(g * torch.ones_like(g)) for g in grads)
        _ = torch.autograd.grad(vT_g, weights, retain_graph=False)

        return False
    except RuntimeError as e:
        if "scaled_dot_product" in str(e).lower():
            return True
        raise e

def _relative_change(values: List[float], prev_mean: float) -> float:
    current_mean = float(np.mean(values))
    return abs(current_mean - prev_mean) / (abs(prev_mean) + 1e-6)

def compute_hessian_trace_hawq(
        model: nn.Module,
        dataloader,
        criterion,
        device: str = "cuda",
        max_batches: int = 10,
        max_iter_per_batch: int = 100,
        tol: float = 1e-3,
        group_by: str = "layer",
        seed: int = 42,
) -> Dict[str, float]:
    # Hutchinson's estimator draws random Rademacher vectors; seeded, the
    # trace (and the HAWQ-V2 ranking) is the same on every run and machine.
    torch.manual_seed(seed)
    model.eval()
    model.to(device)

    target_layers = {
        name: module.weight
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    }

    trace_accumulator: Dict[str, List[float]] = {name: [] for name in target_layers}
    weights = list(target_layers.values())
    names = list(target_layers.keys())

    try:
        test_inputs, test_targets = next(iter(dataloader))
    except StopIteration:
        return {}

    needs_fallback = _check_needs_sdp_fallback(
        model, criterion, test_inputs.to(device), test_targets.to(device), weights
    )

    if needs_fallback:
        print(
            "Detected optimised attention. Activating math fallback for HAWQ..."
        )

    # Main loop
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)

        trace_vhv: Dict[str, List[float]] = {name: [] for name in names}
        trace_prev = {name: 0.0 for name in names}

        for iteration in range(max_iter_per_batch):
            model.zero_grad()

            # Usamos nuestro context manager inteligente
            with _conditional_math_sdp(needs_fallback):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            vs = [
                torch.randint_like(w, high=2, device=device) * 2.0 - 1.0
                for w in weights
            ]

            grads = torch.autograd.grad(loss, weights, create_graph=True)
            vT_g = sum(torch.sum(g * v) for g, v in zip(grads, vs))
            Hv = torch.autograd.grad(vT_g, weights, retain_graph=False)

            # Acumular estimación por capa
            for i, name in enumerate(names):
                avg_trace = torch.sum(vs[i] * Hv[i]).item() / weights[i].numel()
                trace_vhv[name].append(avg_trace)

            # Convergencia...
            if iteration > 0:
                converged = all(
                    _relative_change(trace_vhv[name], trace_prev[name]) < tol
                    for name in names
                )
                if converged:
                    break

            trace_prev = {name: float(np.mean(trace_vhv[name])) for name in names}

        # Batch averages
        for name in names:
            batch_trace = float(np.mean(trace_vhv[name]))
            trace_accumulator[name].append(batch_trace)

        del inputs, targets, outputs, loss, grads, vs, vT_g, Hv

    # Cleanup
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        name: float(np.mean(vals)) if vals else 0.0
        for name, vals in trace_accumulator.items()
    }
