from transformers import AutoImageProcessor
from torch.utils.data import DataLoader, Dataset

def get_data(dataset: Dataset, processor: AutoImageProcessor, train, val, test, batch_size=8):
    train_data = dataset(train, processor)
    val_data = dataset(val, processor)
    test_data = dataset(test, processor)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader