"""
main class for building a DL pipeline.

"""

import click
from batch import Batch
from model.linear import DNN
from model.cnn import VGG16, VGG11
from data import FashionDataset
from utils import Stage
import torch


@click.group()
def cli():
    pass


@cli.command()
def train():
    batch_size = 16
    num_workers = 8

    path = "fashion-mnist_train.csv"
    trainset = FashionDataset(path=path)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    model = VGG11(in_channels=1, num_classes=10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    batch = Batch(
        stage=Stage.TRAIN,
        model=model,
        device=torch.device("cpu"),
        loader=trainloader,
        criterion=criterion,
        optimizer=optimizer,
    )
    batch.run(
        "Run run run run. Run run run away. Oh Oh oH OHHHHHHH yayayayayayayayaya! - David Byrne"
    )


if __name__ == "__main__":
    cli()
