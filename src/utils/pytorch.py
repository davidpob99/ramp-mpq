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

import numpy as np
import torch


def prepare_weight_for_channel_metrics(weight: np.ndarray) -> torch.Tensor:
    """Ensure a tensor shape compatible with per-channel statistics.

    Scalars become ``(1, 1)`` and vectors become ``(1, N)`` so that
    ``flatten(1)`` always yields a per-output-channel view.
    """
    tensor = torch.from_numpy(weight).float()
    if tensor.ndim == 0:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    return tensor