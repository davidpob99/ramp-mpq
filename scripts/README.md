# Scripts
| Script | Function |
| --- | --- |
| `sensitivity_measurement.sh` | Runs the sensitivity flow for every experiment on the full test set. |
| `ramp.sh` | Runs the RAMP flow for the four models on the target device, from the paper's sensitivity tables. |
| `run_all.sh` | Runs sensitivity measurement and then RAMP end to end for one experiment, using its own fresh table. |
| `device.sh` | Sourced by the launch scripts: puts the device in the paper's measurement conditions (CPU governor, priority). |
| `build_quantized.py` | Builds a quantized ONNX model from the FP32 model and a `quantization_policies.json` configuration. |
| `check_dataset.py` | Checks a local SuSy Dataset copy against the official file list. |
| `check_models.sh` | Checks that the local FP32 ONNX models are byte-identical to the paper's. |
| `latency_check.py` | Diagnoses where the FP32 latency goes on a device (isolated vs. flow loop, thermal drift). |
| `profile_layers.py` | Profiles per-layer FP32 vs. INT8 timings on the device and can write a re-profiled sensitivity table. |
