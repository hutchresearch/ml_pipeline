"""
main class for building a DL pipeline.

"""

"""
the main entry point for training a model

coordinates:

- datasets
- dataloaders
- runner

"""
from model.linear import DNN
from model.cnn import VGG16, VGG11
from data.dataset import MnistDataset
from pipeline.utils import Stage
from pipeline.runner import Runner
import torch
from pathlib import Path
from data.collate import channel_to_batch
import hydra
from omegaconf import DictConfig
from accelerate import Accelerator
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pipeline.utils import Stage


@hydra.main(config_path="config", config_name="main", version_base=None)
def train(config: DictConfig):
    accelerator = Accelerator()
    if config.debug:
        breakpoint()
    lr = config.lr
    batch_size = config.batch_size
    num_workers = config.num_workers
    epochs = config.epochs

    train_path = Path(config.app_dir) / "data/mnist_train.csv"
    trainset = MnistDataset(path=train_path)

    dev_path = Path(config.app_dir) / "data/mnist_test.csv"
    devset = MnistDataset(path=dev_path)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # collate_fn=channel_to_batch,
    )
    devloader = torch.utils.data.DataLoader(
        devset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # collate_fn=channel_to_batch,
    )
    model = VGG11(in_channels=1, num_classes=10)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model, optimizer, trainloader = accelerator.prepare(model, optimizer, trainloader)
    train_runner = Runner(
        stage=Stage.TRAIN,
        model=model,
        loader=trainloader,
        loss_function=loss_function,
        optimizer=optimizer,
        accelerator=accelerator,
        config=config,
    )

    for epoch in range(epochs):
        if epoch % config.dev_after == 0:
            dev_log = dev_runner.run("dev epoch")
        else:
            train_log = train_runner.run("train epoch")


if __name__ == "__main__":
    train()
