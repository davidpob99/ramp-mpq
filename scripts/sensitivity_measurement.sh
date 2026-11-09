#!/usr/bin/env bash
# Ground-truth sensitivity sweep: every experiment on the whole test set.
# Hours per model; run it inside tmux or with nohup.
#
# On the Raspberry Pi and the Mac, launch it with sudo -E so the device is put in
# the paper's measurement conditions first (see scripts/device.sh):
#
#   sudo -E ./scripts/sensitivity_measurement.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/device.sh

device_setup "$@"
device_exec "$(device_uv)" run python -m src.flows.sensitivity_measurement.sensitivity_measurement -m \
    'experiment=glob(*)' max_test_batches=null
