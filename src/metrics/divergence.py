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
import torch.nn.functional as F

def compute_batch_kl_mean(
    prediction_logits: torch.Tensor,
    target_logits: torch.Tensor,
) -> float:
    """Compute mean KL(target || prediction) over a batch from logits."""
    log_prob_pred = F.log_softmax(prediction_logits, dim=1)
    prob_target = F.softmax(target_logits, dim=1)
    prob_target = torch.clamp(prob_target, min=1e-12)
    log_prob_target = torch.log(prob_target)
    kl_per_sample = torch.sum(prob_target * (log_prob_target - log_prob_pred), dim=1)
    return float(kl_per_sample.mean().item())

def compute_batch_jsd_mean(
    prediction_logits: torch.Tensor,
    target_logits: torch.Tensor,
) -> float:
    """Compute mean Jensen-Shannon divergence over a batch."""

    log_p = F.log_softmax(target_logits, dim=1)
    log_q = F.log_softmax(prediction_logits, dim=1)

    p = log_p.exp()
    q = log_q.exp()

    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp(min=1e-12))

    kl_p_m = torch.sum(p * (log_p - log_m), dim=1)
    kl_q_m = torch.sum(q * (log_q - log_m), dim=1)

    jsd = 0.5 * (kl_p_m + kl_q_m)

    return float(jsd.mean().item())