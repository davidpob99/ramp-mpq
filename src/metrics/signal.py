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

import torch

def compute_range_ratio(weight: torch.Tensor) -> float:
    """Ratio between the widest and narrowest per-output-channel value range."""
    ranges = weight.flatten(1).max(dim=1).values - weight.flatten(1).min(dim=1).values
    return (ranges.max() / (ranges.min() + 1e-8)).item()

def compute_channel_range_cv(weight: torch.Tensor) -> float:
    """Coefficient of variation of the per-output-channel value ranges."""
    ranges = weight.flatten(1).max(dim=1).values - weight.flatten(1).min(dim=1).values
    return (ranges.std() / (ranges.mean() + 1e-8)).item()

def compute_batch_snr_mean(
    reference: torch.Tensor,
    estimate: torch.Tensor,
) -> float:
    """Compute SNR per sample and return the batch mean in dB."""
    signal_power = torch.mean(reference * reference, dim=1)
    noise_power = torch.mean((reference - estimate) ** 2, dim=1)
    eps = torch.finfo(reference.dtype).eps
    snr_per_sample = 10.0 * torch.log10(
        signal_power / torch.clamp(noise_power, min=eps)
    )
    return float(snr_per_sample.mean().item())

def compute_estimated_sqnr(weight: torch.Tensor, bits: int = 8) -> float:
    """Estimate SQNR (dB) for a uniform quantization step over the tensor range."""
    signal = weight.pow(2).mean()
    q_step = (weight.max() - weight.min()) / (2**bits)
    noise = q_step.pow(2) / 12
    return (10 * torch.log10(signal / (noise + 1e-8))).item()