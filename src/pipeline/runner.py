"""
runner for training and valdating
"""
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pipeline.utils import Stage
from omegaconf import DictConfig
from accelerate import Accelerator


class Runner:
    def __init__(
        self,
        stage: Stage,
        model: nn.Module,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_function: nn.Module,
        accelerator: Accelerator,
        config: DictConfig = None,
    ):
        self.config = config
        self.stage = stage
        self.model = model
        self.loader = loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.loss = 0

    def run(self, desc):
        # set the model to train model
        if self.stage == Stage.TRAIN:
            self.model.train()
        if self.config.debug:
            breakpoint()
        for batch, (x, y) in enumerate(tqdm(self.loader, desc=desc)):
            self.optimizer.zero_grad()
            loss = self._run_batch((x, y))
            self.accelerator.backward(loss)  # Send loss backwards to accumulate gradients
            self.optimizer.step()  # Perform a gradient update on the weights of the mode
            self.loss += loss.item()
        return self.loss

    def _run_batch(self, sample):
        true_x, true_y = sample
        pred_y = self.model(true_x)
        loss = self.loss_function(pred_y, true_y)
        return loss
