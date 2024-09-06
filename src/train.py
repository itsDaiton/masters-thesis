import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def linear_probe(model, train_loader: DataLoader, val_loader: DataLoader, config):
    model.to(config.device)
    
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.backbone.classifier.parameters():
        param.requires_grad = True
        
    optimizer = config.optimizer(model.parameters(), lr=config.lr)
    criterion = config.criterion
    
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_samples = 0
        
        for batch in tqdm(train_loader):
            images, labels = batch
            images, labels = images.to(config.device), labels.to(config.device)    
            optimizer.zero_grad()         
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()       
            train_loss += loss.item()
            train_correct += (predictions == labels).sum().item()
            train_samples += labels.size(0)
        
        train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_samples

        model.eval()
           
        val_loss = 0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                images, labels = images.to(config.device), labels.to(config.device)
                
                outputs = model(images)
                _, predictions = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_correct += (predictions == labels).sum().item()
                val_samples += labels.size(0)
        
        val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_samples
        
        print(f"Epochs: {epoch + 1}/{config.num_epochs} 
              | train_loss: {train_loss:.4f} 
              | train_acc: {train_accuracy:.4f} 
              | val_loss: {val_loss:.4f} 
              | val_acc: {val_accuracy:.4f}")