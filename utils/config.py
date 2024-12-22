import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss


class SchedulerConfig:
    def __init__(self, enabled=False, warmup_epochs=0, eta_min=0):
        self.enabled = enabled
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min

    def get_scheduler_settings(self):
        return {attr: value for attr, value in vars(self).items()}


class GradientClippingConfig:
    def __init__(self, enabled=False, max_norm=0):
        self.enabled = enabled
        self.max_norm = max_norm

    def get_clipping_settings(self):
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
        gradient_clipping_config=GradientClippingConfig(),
        scheduler_config=SchedulerConfig(),
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.criterion = criterion
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.gradient_clipping_config = gradient_clipping_config
        self.scheduler_config = scheduler_config

    def get_settings(self):
        settings = {attr: value for attr, value in vars(self).items()}
        settings["gradient_clipping_config"] = (
            self.gradient_clipping_config.get_clipping_settings()
        )
        settings["scheduler_config"] = self.scheduler_config.get_scheduler_settings()
        return settings
