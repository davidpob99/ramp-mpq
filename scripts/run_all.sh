#!/usr/bin/env bash
#
# Copyright 2026 David Población Criado
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#

# scripts/run_all.sh <experiment>
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/device.sh

EXPERIMENT="${1:?experiment, e.g. timm_resnet18_susy}"
RUN_DIR="$PWD/outputs/all/${EXPERIMENT}/$(date +%Y%m%d-%H%M%S)"
case "$DEVICE_KIND" in
    rpi) THREADS=4 ;;
    *)   THREADS=8 ;;
esac
device_setup "$@"
UV="$(device_uv)"

device_exec "$UV" run python -m src.flows.sensitivity_measurement.sensitivity_measurement \
    "experiment=${EXPERIMENT}" max_test_batches=null \
    "hydra.run.dir=${RUN_DIR}/sensitivity"

device_exec "$UV" run python -m src.flows.ramp.ramp \
    "experiment=${EXPERIMENT}" max_test_batches=null \
    "sensitivity_table=${RUN_DIR}/sensitivity/sensitivity_layer_table.csv" \
    "session_options.intra_op_num_threads=${THREADS}" \
    "hydra.run.dir=${RUN_DIR}/ramp"