from tkinter import W
import torch
import wandb
import numpy as np
from PIL import Image
from einops import rearrange
from typing import Protocol, Tuple, Optional


class Logger(Protocol):
    def metrics(self, metrics: dict, epoch: int):
        """loss etc."""

    def hyperparameters(self, hyperparameters: dict):
        """model states"""

    def predictions(self, predictions: dict):
        """inference time stuff"""

    def images(self, images: np.ndarray):
        """log images"""


class WandbLogger:
    def __init__(self, project: str, entity: str, name: Optional[str], notes: str):
        self.project = project
        self.entity = entity
        self.notes = notes
        self.experiment = wandb.init(project=project, entity=entity, notes=notes)
        self.experiment.name = name

        self.data_dict = {}

    def metrics(self, metrics: dict):
        """loss etc."""

        self.data_dict.update(metrics)

    def hyperparameters(self, hyperparameters: dict):
        """model states"""
        self.experiment.config.update(hyperparameters, allow_val_change=True)

    def predictions(self, predictions: dict):
        """inference time stuff"""

    def image(self, image: dict):
        """log images to wandb"""
        self.data_dict.update({"Generate Image": image})

    def video(self, images: str, title: str):
        """log images to wandb"""

        images = np.uint8(rearrange(images, "t b c h w -> b t c h w"))
        self.data_dict.update({f"{title}": wandb.Video(images, fps=20)})

    def flush(self):
        self.experiment.log(self.data_dict)
        self.data_dict = {}


class DebugLogger:
    def __init__(self, project: str, entity: str, name: str, notes: str):
        self.project = project
        self.entity = entity
        self.name = name
        self.notes = notes

    def metrics(self, metrics: dict, epoch: int = None):
        """
        loss etc.
        """
        print(f"metrics: {metrics}")

    def hyperparameters(self, hyperparameters: dict):
        """
        model states
        """
        print(f"hyperparameters: {hyperparameters}")

    def predictions(self, predictions: dict):
        """
        inference time stuff
        """


class Checkpoint:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path

    def load(self) -> Tuple:
        checkpoint = torch.load(self.checkpoint_path)
        model = checkpoint["model"]
        optimizer = checkpoint["optimizer"]
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        return (model, optimizer, epoch, loss)

    def save(self, model: torch.nn.Module, optimizer, epoch, loss):
        checkpoint = {
            "model": model,
            "optimizer": optimizer,
            "epoch": epoch,
            "loss": loss,
        }
        import random
        import string

        name = "".join(random.choices(string.ascii_letters, k=10)) + ".tar"
        torch.save(checkpoint, f"{name}")
