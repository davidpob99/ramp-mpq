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

import os
from typing import Any, List, Optional

import lightning as L
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        # Get original output (image and target)
        original_tuple = super().__getitem__(index)

        # Get the image path
        path = self.imgs[index][0]  # `imgs` contains (path, class_index)

        # Return image, target, and path
        return original_tuple + (path,)


class ImageFolderDataModule(L.LightningDataModule):
    def __init__(
            self,
            root_train: str,
            root_test: str,
            batch: int,
            root_val: Optional[str] = None,
            transform_train: transforms.Compose = None,
            transform_val: transforms.Compose = None,
            transform_test: transforms.Compose = None,
            val_size: Optional[int] = 0.15,
            num_workers: int = 0,
            pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train: Optional[Dataset] = None
        self.val: Optional[Dataset] = None
        self.test: Optional[Dataset] = None

    @property
    def class_names(self) -> List[str]:
        """Get a list with the name of each class

        Returns:
            List[str]: class names
        """
        if self.hparams.get("root_train"):
            classes_train = [
                d
                for d in os.listdir(self.hparams.root_train)
                if os.path.isdir(os.path.join(self.hparams.root_train, d))
            ]
        else:
            classes_train = []
        if self.hparams.get("root_test"):
            classes_test = [
                d
                for d in os.listdir(self.hparams.root_test)
                if os.path.isdir(os.path.join(self.hparams.root_test, d))
            ]
        else:
            classes_test = []

        return sorted(list(set(classes_train + classes_test)))

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def setup(self, stage: str) -> None:
        if self.hparams.root_val:  # If a validation set is provided
            self.train = ImageFolderWithPaths(
                root=self.hparams.root_train,
                transform=self.hparams.transform_train,
                allow_empty=True,
            )
            self.val = ImageFolderWithPaths(
                root=self.hparams.root_val,
                transform=self.hparams.transform_val,
                allow_empty=True,
            )

        else:
            dataset = ImageFolderWithPaths(
                root=self.hparams.root_train,
                transform=self.hparams.transform_train,
                allow_empty=True,
            )
            # Compute the sizes of the train and validation sets
            total_size = len(dataset)
            train_size = int(total_size * (1 - self.hparams.val_size))
            val_size = total_size - train_size
            self.train, self.val = random_split(dataset, [train_size, val_size])
        self.test = ImageFolderWithPaths(
            root=self.hparams.root_test,
            transform=self.hparams.transform_test,
            allow_empty=True,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.train,
            batch_size=self.hparams.batch,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.val,
            batch_size=self.hparams.batch,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.test,
            batch_size=self.hparams.batch,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
