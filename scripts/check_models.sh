#!/usr/bin/env bash
# Check that this machine's FP32 ONNX models are byte-identical to the reference
# ones (the checkpoints the paper was measured with). Run it on every device:
#
#   ./scripts/check_models.sh
#
# Hashes both the graph (model.onnx) and its external weights (<uuid>.data):
# the graph alone can match while the weights differ. Exits 1 on any mismatch.
set -uo pipefail
cd "$(dirname "$0")/.."

if command -v sha256sum >/dev/null; then
    sha() { sha256sum "$1" | cut -d' ' -f1; }
else
    sha() { shasum -a 256 "$1" | cut -d' ' -f1; }  # macOS
fi

DIR=models/susy-classification/fp32
REFERENCE="
c8c70bc6fb7318f0cb787d11f1bf203ff40ea351271c9ffe10398ebfca25e09b timm_convnext_tiny/4cc0ce04-10bd-11f1-9879-11cc9b8f2550.data
e7803fc044b2089b813dd7972ddd96d0aeedbe3cc2b572dcf59c0d84c0820ddd timm_convnext_tiny/model.onnx
ca54f7cc3a3f77cd5b2776bd880c4e0d206a6af00b59ce80b80aa6842e3273ed timm_efficientnet-b0/c2c7636e-10d2-11f1-82ff-5dc2dc96617f.data
b218dd842d626d6dea78d224adc015fbdd5de6ab1de61eb079d6ca2953c9a190 timm_efficientnet-b0/model.onnx
4b110b082586347656c97c6703dff5a4d8a1b0b4f4905d5cf882ffb4be0f56fe timm_resnet18/c6d4102e-10f5-11f1-8188-61f6185d7076.data
f2d55541757c82c10bc4022fbdf26fc775ce1a84999ff190ae6943baea99982a timm_resnet18/model.onnx
ff5e53faf120758087602a0c50bd80b8ca2e5107dce3e8a92a1ec6ddda00c62a timm_tinyvit/faedb7c2-2156-11f1-a320-3b30509f0825.data
cbf7338c3d896d203b770b60e5fbd8ba891556e3eefc49de5e7d41eadfe5673e timm_tinyvit/model.onnx
"

failed=0
while read -r expected file; do
    [[ -z "$file" ]] && continue
    path="$DIR/$file"
    if [[ ! -f "$path" ]]; then
        echo "MISSING   $path"; failed=1; continue
    fi
    actual=$(sha "$path")
    if [[ "$actual" == "$expected" ]]; then
        echo "OK        $path"
    else
        echo "DIFFERENT $path"; failed=1
    fi
done <<<"$REFERENCE"

# Anything besides the reference files (a stale .data from another export)
# would not be loaded, but is worth knowing about.
find "$DIR" -type f \( -name '*.onnx' -o -name '*.data' \) | sort | while read -r path; do
    grep -q " ${path#"$DIR"/}$" <<<"$REFERENCE" || echo "EXTRA     $path"
done

((failed)) && { echo "Models differ from the reference." >&2; exit 1; }
echo "All models match the reference."
