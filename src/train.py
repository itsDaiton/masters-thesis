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
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_accuracy = train_correct / train_samples

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
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = val_correct / val_samples
        
        print(f"Epochs: {epoch + 1}/{config.num_epochs} 
              | train_loss: {avg_train_loss:.4f} 
              | train_acc: {avg_train_accuracy:.4f} 
              | val_loss: {avg_val_loss:.4f} 
              | val_acc: {avg_val_accuracy:.4f}")

def evaluate(model, test_loader: DataLoader, config):  
    model.to(config.device)
    model.eval()

    criterion = config.criterion
    
    test_loss = 0
    test_correct = 0
    test_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            images, labels = batch
            images, labels = images.to(config.device), labels.to(config.device)
            
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            ttest_correct += (predictions == labels).sum().item()
            test_samples += labels.size(0)
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_accuracy = test_correct / test_samples

    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}")