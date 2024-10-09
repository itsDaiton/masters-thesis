import torch
import torch.nn as nn
from torch.optim import Adam

class Config:
    def __init__(self, batch_size=8, lr=5e-5, num_epochs=5, optimizer=Adam, binary_task=False, weight_decay=1e-4):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.binary_task = binary_task
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        
    def get_loss_function(self):
        if self.binary_task:
            return nn.BCEWithLogitsLoss()
        else:
            return nn.CrossEntropyLoss()