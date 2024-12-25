import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss


class SchedulerConfig:
    def __init__(
        self,
        enabled=False,
        warmup_epochs=0,
        eta_min=0,
        linear_start_factor=0.1,
        linear_end_factor=1.0,
    ):
        self.enabled = enabled
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        self.linear_start_factor = linear_start_factor
        self.linear_end_factor = linear_end_factor

    def get_scheduler_settings(self):
        return {attr: value for attr, value in vars(self).items()}


class GradientClippingConfig:
    def __init__(self, enabled=False, max_norm=0):
        self.enabled = enabled
        self.max_norm = max_norm

    def get_clipping_settings(self):
        return {attr: value for attr, value in vars(self).items()}


class EarlyStoppingConfig:
    def __init__(self, enabled=False, patience=3, delta=0):
        self.enabled = enabled
        self.patience = patience
        self.delta = delta

    def get_early_stopping_settings(self):
        return {attr: value for attr, value in vars(self).items()}


class Config:
    def __init__(
        self,
        batch_size=8,
        lr=5e-5,
        num_epochs=5,
        optimizer=Adam,
        criterion=CrossEntropyLoss(),
        weight_decay=1e-4,
        early_stopping=EarlyStoppingConfig(),
        gradient_clipping=GradientClippingConfig(),
        scheduler=SchedulerConfig(),
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.criterion = criterion
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.gradient_clipping = gradient_clipping
        self.scheduler = scheduler

    def get_settings(self):
        settings = {attr: value for attr, value in vars(self).items()}
        settings["early_stopping"] = self.early_stopping.get_early_stopping_settings()
        settings["gradient_clipping"] = self.gradient_clipping.get_clipping_settings()
        settings["scheduler"] = self.scheduler.get_scheduler_settings()
        return settings


config = Config()
print(config.get_settings())
