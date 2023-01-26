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
from pipeline.runner import Runner
from model.linear import DNN
from model.cnn import VGG16, VGG11
from data.dataset import MnistDataset
from pipeline.utils import Stage
import torch
from pathlib import Path
from data.collate import channel_to_batch
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="main")
def train(config: DictConfig):
    if config.debug:
        breakpoint()
    lr = config.lr
    batch_size = config.batch_size
    num_workers = config.num_workers
    device = config.device
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
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_runner = Runner(
        stage=Stage.TRAIN,
        model=model,
        device=torch.device(device),
        loader=trainloader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
    )
    dev_runner = Runner(
        stage=Stage.DEV,
        model=model,
        device=torch.device(device),
        loader=devloader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
    )

    for epoch in range(epochs):
        if epoch % config.dev_after == 0:
            dev_log = dev_runner.run("dev epoch")
        else:
            train_log = train_runner.run("train epoch")


if __name__ == "__main__":
    train()
