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

import io
import time
from typing import Any, Dict, Mapping, Tuple, Union

import lightning as L
import numpy as np
import timm
import torch
from timm.data import resolve_data_config
from torchmetrics import MaxMetric, MeanMetric, MetricCollection
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall

ORT_TYPE_NP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(double)": np.float64,
}


class TimmClassification(L.LightningModule):
    def __init__(
            self,
            model_name,
            nc,
            optimizer,
            scheduler,
            weightsfile=None,
            warmup_epochs: int = 0,
            warmup_start_factor: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = timm.create_model(model_name, pretrained=True, num_classes=nc)
        self.config = resolve_data_config(self.model.pretrained_cfg, model=self.model)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_metric = MetricCollection(
            {
                "acc": Accuracy(task="multiclass", num_classes=nc),
                "precision": Precision(
                    task="multiclass", average="macro", num_classes=nc
                ),
                "recall": Recall(task="multiclass", average="macro", num_classes=nc),
                "f1": F1Score(task="multiclass", average="macro", num_classes=nc),
            }
        )
        self.val_metric = MetricCollection(
            {
                "acc": Accuracy(task="multiclass", num_classes=nc),
                "precision": Precision(
                    task="multiclass", average="macro", num_classes=nc
                ),
                "recall": Recall(task="multiclass", average="macro", num_classes=nc),
                "f1": F1Score(task="multiclass", average="macro", num_classes=nc),
            }
        )
        self.test_metric = MetricCollection(
            {
                "acc": Accuracy(task="multiclass", num_classes=nc),
                "precision": Precision(
                    task="multiclass", average="macro", num_classes=nc
                ),
                "recall": Recall(task="multiclass", average="macro", num_classes=nc),
                "f1": F1Score(task="multiclass", average="macro", num_classes=nc),
                # "confmat": ConfusionMatrix(task="multiclass", num_classes=nc)
            }
        )

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_metric_best = {name: MaxMetric() for name in self.val_metric.keys()}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.model`"""
        return self.model(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_metric.reset()
        for k, v in self.val_metric_best.items():
            v.reset()

    def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Union[torch.Tensor, Mapping[str, Any], None]:
        images, targets, _ = batch
        output = self.model(images)
        loss = self.criterion(output, targets)

        # update and log metrics
        self.train_loss(loss)
        self.train_metric(output, targets)

        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def on_train_epoch_end(self) -> None:
        metrics = self.train_metric.compute()
        loss = self.train_loss.compute()
        for k, v in metrics.items():
            self.log(f"train/{k}", v, sync_dist=True, prog_bar=True)
        self.log("train/loss_epoch", loss, sync_dist=True, prog_bar=False)
        self.train_metric.reset()
        self.train_loss.reset()

    def validation_step(
            self, batch, batch_idx
    ) -> Union[torch.Tensor, Mapping[str, Any], None]:
        images, targets, _ = batch
        output = self.model(images)
        loss = self.criterion(output, targets)

        # update and log metrics
        self.val_loss(loss)
        self.val_metric(output, targets)

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        metrics = self.val_metric.compute()
        loss = self.val_loss.compute()
        self.log("val/loss_epoch", loss, sync_dist=True, prog_bar=False)
        for name, value in metrics.items():
            self.val_metric_best[name](value)
            self.log(f"val/{name}", value, sync_dist=True)
            self.log(
                f"val/{name}_best", self.val_metric_best[name].compute(), sync_dist=True
            )

        self.val_metric.reset()
        self.val_loss.reset()

    def test_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict:
        """Perform a single test step on a batch of data from the test set."""
        images, targets, filenames = batch
        output = self.model(images)
        loss = self.criterion(output, targets)

        # update and log metrics
        self.test_loss(loss)
        self.test_metric(output, targets)

        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {
            "loss": loss,
            "output": output,
            "filenames": filenames,
        }

    def test_polygraphy_step(self, batch, runner) -> Tuple[torch.Tensor, float]:
        images, targets, filenames = batch
        ta = time.time()
        outputs = runner.infer({"input": images}, check_inputs=False)
        tb = time.time()
        outputs = torch.tensor(outputs["output"].reshape(-1, self.hparams.nc))
        self.test_metric.update(outputs, targets)

        return outputs, tb - ta

    def test_onnxruntime_step(
            self, batch, session, input_name, output_name
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        images, targets, filenames = batch

        if isinstance(images, torch.Tensor):
            images_np = images.numpy()
        else:
            images_np = images

        if not images_np.flags["C_CONTIGUOUS"]:
            images_np = np.ascontiguousarray(images_np)

        expected_ort_type_str = session.get_inputs()[0].type
        target_dtype = ORT_TYPE_NP.get(expected_ort_type_str)

        if images_np.dtype != target_dtype:
            images_np = images_np.astype(target_dtype)

        ta = time.perf_counter()
        outputs = session.run([output_name], {input_name: images_np})
        tb = time.perf_counter()

        outputs_tensor = torch.tensor(outputs[0])
        if outputs_tensor.dim() > 2:
            outputs_tensor = outputs_tensor.reshape(-1, self.hparams.nc)

        self.test_metric.update(outputs_tensor, targets)

        return outputs_tensor, targets, (tb - ta)

    def test_tensorrt_step(
            self, batch, trt_engine, context, buffers, datamodule
    ) -> float:
        raise NotImplementedError()

    def on_test_epoch_end(self) -> None:
        "Lightning hook that is called when a test epoch ends."
        metrics = self.test_metric.compute()
        for k, v in metrics.items():
            self.log(f"test/{k}", v, sync_dist=True, prog_bar=True)
        self.test_metric.reset()
        self.test_loss.reset()

    def to_onnx(self, input_sample=None, **kwargs) -> bytes:
        """Returns the model in ONNX format

        Args:
            input_sample: An input for tracing. Default: None (Use self.example_input_array)
            **kwargs: Will be passed to torch.onnx.export function.

        Returns:
            str: file path of the exported ONNX model
        """
        if input_sample is None:
            input_sample = torch.zeros(
                self.trainer.datamodule.hparams.batch,
                self.config["input_size"][0],
                self.config["input_size"][1],
                self.config["input_size"][2],
            )
        f = io.BytesIO()
        torch.onnx.export(self, input_sample, f, **kwargs)
        onnx_model = f.getvalue()

        return onnx_model

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization."""
        optimizer = self.hparams.optimizer(params=self.parameters())

        warmup_epochs = int(getattr(self.hparams, "warmup_epochs", 0) or 0)
        warmup_start_factor = float(
            getattr(self.hparams, "warmup_start_factor", 0.1) or 0.1
        )

        base_scheduler = (
            self.hparams.scheduler(optimizer=optimizer)
            if self.hparams.scheduler is not None
            else None
        )

        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )

            if base_scheduler is not None:
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, base_scheduler],
                    milestones=[warmup_epochs],
                )
            else:
                scheduler = warmup_scheduler

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        if base_scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": base_scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return {"optimizer": optimizer}

    @staticmethod
    def _get_weights(model: torch.nn.Module, weightsfile: str):
        """
        Load weights into the model.
        Args:
            model (torch.nn.Module): The model to load weights into.
            params (EasyDict): The parameters containing the weights file path and model type.
        """
        extension = weightsfile.split(".")[-1]
        if extension == "pth" or extension == "pt":
            state_dict = torch.load(weightsfile)
            model.load_state_dict(state_dict)
        else:
            raise Exception(f"Weights with extension .{extension} are not supported")
