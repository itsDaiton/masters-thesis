import torch
import torch.nn as nn

class Config:
    """ Configuration class for training hyperparameters. """
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 8
        self.lr = 5e-5
        self.num_epochs = 2
        self.optimizer = torch.optim.AdamW
        self.criterion = nn.CrossEntropyLoss()