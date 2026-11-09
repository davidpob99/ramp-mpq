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

import logging
import random
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageFile
from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.quantization.registry import QDQRegistry, QLinearOpsRegistry

SUPPORTED_QUANT_OP_TYPES: set[str] = set(QDQRegistry).union(QLinearOpsRegistry)


class CalibrationDataReaderFromFolder(CalibrationDataReader):
    def __init__(
            self,
            folder_path: str,
            size: Tuple[int, int],
            input_name: str = "input",
            max_images: int = 1000,
            seed: int = 42,
    ):
        self.folder_path = folder_path
        self.size = size
        self.input_name = input_name
        self.max_images = max_images
        self.seed = seed
        self.data_cache = None
        self.index = 0

    def get_next(self):
        if self.data_cache is None:
            self.prepare_data()

        if self.index < len(self.data_cache):
            data = self.data_cache[self.index]
            self.index += 1
            return {self.input_name: data}
        else:
            return None

    def prepare_data(self):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        _extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp")
        # Sorted first: rglob order depends on the filesystem, so without it
        # the same seed would still pick different images on another machine.
        filenames = sorted(
            str(file)
            for file in Path(self.folder_path).rglob("*")
            if file.suffix.lower() in _extensions
        )

        # Own RNG rather than the global one, so the subset does not depend on
        # whatever else consumed `random` earlier in the process. The same seed
        # always calibrates on the same images, and MinMax ranges are then
        # fully reproducible.
        random.Random(self.seed).shuffle(filenames)

        logging.info(
            f"Found {len(filenames)} images for calibration, using max "
            f"{self.max_images} (seed {self.seed})"
        )

        self.data_cache = []
        processed_count = 0

        for filename in filenames:
            if processed_count >= self.max_images:
                break

            try:
                img = Image.open(filename)
                # Convert to RGB to ensure 3 channels
                if img.mode != "RGB":
                    img = img.convert("RGB")
                # Resize the image
                resized_img = img.resize(self.size)
                # Convert to NumPy array with desired shape (1, 3, H, W)
                resized_np_array = (
                        np.array(resized_img, dtype=np.float32) / 255.0
                )  # Normalize to [0, 1] range
                resized_np_array = np.transpose(
                    resized_np_array, (2, 0, 1)
                )  # from (H, W, C) to (C, H, W)
                # Add batch dimension
                resized_np_array = np.expand_dims(resized_np_array, axis=0)

                self.data_cache.append(resized_np_array)
                processed_count += 1

            except Exception as e:
                logging.warning(f"Failed to process image {filename}: {e}")
                continue

        logging.info(f"Prepared {len(self.data_cache)} calibration images")
