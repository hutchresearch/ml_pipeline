import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from data import FashionDataset
from tqdm import tqdm
from utils import Stage


class Batch:
    def __init__(
        self,
        stage: Stage,
        model: nn.Module,
        device,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ):
        """todo"""
        self.stage = stage
        self.device = device
        self.model = model.to(device)
        self.loader = loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss = 0

    def run(self, desc):
        self.model.train()
        epoch = 0
        for epoch, (x, y) in enumerate(tqdm(self.loader, desc=desc)):
            self.optimizer.zero_grad()
            loss = self._run_batch((x, y))
            loss.backward()  # Send loss backwards to accumulate gradients
            self.optimizer.step()  # Perform a gradient update on the weights of the mode
            self.loss += loss.item()

    def _run_batch(self, sample):
        true_x, true_y = sample
        true_x, true_y = true_x.to(self.device), true_y.to(self.device)
        pred_y = self.model(true_x)
        loss = self.criterion(pred_y, true_y)
        return loss


def main():
    model = nn.Conv2d(1, 64, 3)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    path = "fashion-mnist_train.csv"
    dataset = FashionDataset(path)
    batch_size = 16
    num_workers = 1
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    batch = Batch(
        Stage.TRAIN,
        device=torch.device("cpu"),
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loader=loader,
    )
    batch.run("test")


if __name__ == "__main__":
    main()
