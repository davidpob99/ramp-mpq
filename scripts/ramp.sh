#!/usr/bin/env bash
# RAMP final results on the target device, from the paper's sensitivity tables.
#
#   ./scripts/ramp.sh               # Raspberry Pi 5 (Cortex-A76), the default
#   ./scripts/ramp.sh apple-m1      # Apple M1
#
# Run it ON the device the table comes from: the hardware-aware pool is chosen
# from that device's per-node latencies (dur_s), and every configuration's
# latency is measured on the machine running this script. Mixing them selects
# layers by one CPU's speedups and measures another.
#
# Each model is a full test-set pass per configuration (~25 per model), so
# expect hours per model. Run it inside tmux or with nohup.
#
# On the Raspberry Pi and the Mac, launch it with sudo -E so the device is put in
# the paper's measurement conditions first (see scripts/device.sh):
#
#   sudo -E ./scripts/ramp.sh
#
# ONNX Runtime intra-op threads follow the device's cores. The paper used 8 on
# both. On the 4-core Raspberry Pi 5 that oversubscribes the CPU: with the
# current kernel the FP32 ResNet-18 takes 128 ms with 8 threads and 44 ms with
# 4. The Apple M1 has 8 cores and is faster with 8 (ConvNeXt-Tiny FP32: 43 ms
# against 50 ms with 4).
#
# A model that fails does not stop the others; failures are listed at the end.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/device.sh

HW="${1:-cortex-a76}"
case "$HW" in
    cortex-a76) THREADS=4 ;;
    *)          THREADS=8 ;;
esac
TABLES="$PWD/data/paper/sensitivity_tables"
FAILED=()
device_setup "$@"
UV="$(device_uv)"

run() {
    local experiment="$1" table="$2"
    echo "=== ${experiment} on ${HW} ==="
    device_exec "$UV" run python -m src.flows.ramp.ramp \
        "experiment=${experiment}" \
        "sensitivity_table=${TABLES}/${table}__${HW}.csv" \
        max_test_batches=null \
        "session_options.intra_op_num_threads=${THREADS}" \
        || FAILED+=("${experiment}")
}

run timm_resnet18_susy        resnet18
run timm_convnext_tiny_susy   convnext_tiny
run timm_efficientnet-b0_susy efficientnet-b0
run timm_tinyvit_susy         tinyvit

if ((${#FAILED[@]})); then
    echo "Failed: ${FAILED[*]}" >&2
    exit 1
fi
