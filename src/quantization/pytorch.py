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


def get_l2_quantization_error(weight_tensor: torch.Tensor, bits: int = 8) -> float:
    """
    Simulate symetric Min-Max quantization and returns L2 squared perturbation
    """
    w_min, w_max = weight_tensor.min(), weight_tensor.max()

    # Calcular el rango de cuantización dinámicamente según los bits
    q_max = (2 ** (bits - 1)) - 1
    q_min = -q_max

    # Encontrar el valor absoluto máximo para el escalado simétrico
    max_abs_val = max(abs(w_min), abs(w_max))

    # Evitar división por cero si el tensor está completamente a 0
    if max_abs_val == 0:
        return 0.0

    scale = max_abs_val / float(q_max)

    # Simular Q(W): Dividir por escala, redondear y saturar (clipping)
    quantized_weight = torch.round(weight_tensor / scale)
    quantized_weight = torch.clamp(quantized_weight, q_min, q_max)

    # Simular Dequantize(Q(W))
    dequantized_weight = quantized_weight * scale

    # Calcular ||Q(W) - W||_2^2
    l2_error = torch.norm(dequantized_weight - weight_tensor, p=2) ** 2

    return l2_error.item()