import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss


class Config:
    def __init__(
        self,
        batch_size=8,
        lr=5e-5,
        num_epochs=5,
        optimizer=Adam,
        criterion=CrossEntropyLoss(),
        weight_decay=1e-4,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.criterion = criterion
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.weight_decay = weight_decay

    def get_settings(self):
        return {attr: value for attr, value in vars(self).items()}
