"""
main class for building a DL pipeline.

"""

from batch import Batch
from model.linear import DNN
from model.cnn import VGG16, VGG11
from data import MnistDataset
from utils import Stage
import torch
from pathlib import Path
from collate import channel_to_batch

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

    path = Path(config.app_dir) / "storage/mnist_train.csv"
    trainset = MnistDataset(path=path)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # collate_fn=channel_to_batch,
    )
    model = VGG11(in_channels=1, num_classes=10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    batch = Batch(
        stage=Stage.TRAIN,
        model=model,
        device=torch.device(device),
        loader=trainloader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
    )
    log = batch.run(
        "Run run run run. Run run run away. Oh Oh oH OHHHHHHH yayayayayayayayaya! - David Byrne"
    )


if __name__ == "__main__":
    train()
