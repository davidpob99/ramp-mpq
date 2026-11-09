"""Where does the FP32 latency go? A quick check to run on the device.

Measures the raw FP32 model of one experiment three ways, with the flow's
session options (8 intra-op / 1 inter-op threads):

  isolated   session.run on one random input, nothing else in between;
  flow       the flow's own loop (evaluate_on_test_set): each image loaded,
             decoded and scored between two inferences, as the paper did;
  flow-1thr  the same loop with PyTorch limited to one thread.

If "isolated" already matches the paper, the model and ORT are fine and the gap
comes from what runs around the inference. If "flow-1thr" then recovers it,
PyTorch's thread pool is competing with ORT's for the cores. If even
"isolated" is slow, the process itself is being throttled: how it was launched
(sudo, nice, SSH, a background session), the kernel, low power mode or
temperature.

    uv run python scripts/latency_check.py                        # EfficientNet-B0
    uv run python scripts/latency_check.py experiment=timm_resnet18_susy
    uv run python scripts/latency_check.py experiment=timm_resnet18_susy \\
        session_options.intra_op_num_threads=4

    # Does latency drift under sustained load (thermal throttling)? Runs the
    # isolated loop for that many seconds and prints the median of each minute.
    SUSTAIN=600 uv run python scripts/latency_check.py experiment=timm_convnext_tiny_susy
"""

import os
import sys
import time

import numpy as np
import onnxruntime as ort
import rootutils
import torch
from hydra import compose, initialize_config_dir

PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.quantization.evaluation import (  # noqa: E402
    EvaluationContext,
    _model_input_shape,
    evaluate_on_test_set,
    get_session_options,
)

N_IMAGES = 500
N_ISOLATED = 300


def main() -> None:
    overrides = sys.argv[1:] or ["experiment=timm_efficientnet-b0_susy"]
    with initialize_config_dir(str(PROJECT_ROOT / "config"), version_base="1.3"):
        config = compose("ramp.yaml", overrides=["sensitivity_table=unused", *overrides])
    model_path = config.onnx_file
    options = get_session_options(config.session_options)
    print(
        f"{config.model.model_name}, onnxruntime {ort.__version__}, "
        f"intra-op threads {config.session_options.intra_op_num_threads}"
    )

    session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    x = np.random.randn(1, *_model_input_shape(model_path)[1:]).astype(np.float32)
    for _ in range(10):
        session.run(None, {name: x})
    times = []
    for _ in range(N_ISOLATED):
        start = time.perf_counter()
        session.run(None, {name: x})
        times.append(time.perf_counter() - start)
    print(f"isolated   median {np.median(times) * 1000:7.2f} ms")

    sustain = float(os.environ.get("SUSTAIN", 0))
    if sustain:
        begin = minute_start = time.perf_counter()
        minute = []
        while time.perf_counter() - begin < sustain:
            start = time.perf_counter()
            session.run(None, {name: x})
            end = time.perf_counter()
            minute.append(end - start)
            if end - minute_start >= 60:
                elapsed = int((end - begin) // 60)
                print(f"sustained  minute {elapsed:2d}: median {np.median(minute) * 1000:7.2f} ms")
                minute, minute_start = [], end
        return
    del session

    ctx = EvaluationContext(
        preprocessed_path="", session_options=options, output_names=[],
        quantization_config=config.quantization, tensors_range=None, batch_inputs=[],
        batch_labels=[], batch_outputs_fp32=[], num_classes=6, acc_fp32=float("nan"),
        latency_fp32=float("nan"), warmup_runs=5, onnx_file=model_path,
        data_config=config.data, max_test_batches=N_IMAGES,
    )
    _, latency = evaluate_on_test_set(model_path, ctx)
    print(f"flow       median {latency * 1000:7.2f} ms  (torch threads: {torch.get_num_threads()})")

    torch.set_num_threads(1)
    _, latency = evaluate_on_test_set(model_path, ctx)
    print(f"flow-1thr  median {latency * 1000:7.2f} ms")


if __name__ == "__main__":
    main()
