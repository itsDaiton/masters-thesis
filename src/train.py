import torch
from tqdm import tqdm
from utils.models_utils import get_last_layer
from utils.train_utils import print_training_results, print_evaluation_results, print_zero_shot_results, calculate_hard_distillation
from torch.utils.data import DataLoader
    
def train_model(model, train_loader, val_loader, config, architecture, fine_tune=True, with_distillation=False, teacher=None):
    model.to(config.device)
    
    if with_distillation and teacher is not None:   
        teacher.to(config.device)
        teacher.eval()
        
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False
        for param in get_last_layer(model, architecture).parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True        
            
    optimizer = config.optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    criterion = config.criterion
    
    for epoch in range(config.num_epochs):
        model.train()
        
        train_loss = 0
        train_correct = 0
        train_samples = 0
        
        for batch in tqdm(train_loader, desc='Train'):
            images, labels = batch
            images, labels = images.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            
            if with_distillation and teacher is not None:
                with torch._no_grad():
                    teacher_outputs = teacher(images)
                    _, teacher_predictions = torch.max(teacher_outputs, 1)  
                    
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            
            if with_distillation and teacher is not None:
                loss = calculate_hard_distillation(outputs, teacher_predictions, labels, criterion)
            else:
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
            for batch in tqdm(val_loader, desc='Val'):
                images, labels = batch
                images, labels = images.to(config.device), labels.to(config.device)
                
                if with_distillation and teacher is not None:
                    teacher_outputs = teacher(images)
                    _, teacher_predictions = torch.max(teacher_outputs, 1)
                    
                outputs = model(images)
                _, predictions = torch.max(outputs, 1)
                
                if with_distillation and teacher is not None:
                    loss = calculate_hard_distillation(outputs, teacher_predictions, labels, criterion)
                else:
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item()
                val_correct += (predictions == labels).sum().item()
                val_samples += labels.size(0) 
                                      
            avg_val_loss = val_loss / len(val_loader)
            avg_val_accuracy = val_correct / val_samples 
            
            print_training_results(epoch, config.num_epochs, avg_train_loss, avg_train_accuracy, avg_val_loss, avg_val_accuracy) 

def evaluate_model(model, data, config, zero_shot=False):
    if zero_shot:
        tokenized_captions = data.get_tokenized_captions()
        input_ids = tokenized_captions['input_ids'].to(config.device)
        
    dataloader = DataLoader(data, batch_size=config.batch_size) 
    criterion = config.criterion
    
    model.to(config.device)
    model.eval()
    
    test_loss = 0
    test_correct = 0
    test_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Test' if not zero_shot else 'Zero-shot'):
            images, labels = batch
            images, labels = images.to(config.device), labels.to(config.device)
            
            if zero_shot:
                outputs = model(images=images, texts=input_ids)
            else:
                outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            test_correct += (predictions == labels).sum().item()
            test_samples += labels.size(0)
    
    avg_test_loss = test_loss / len(dataloader)
    avg_test_accuracy = test_correct / test_samples
    
    if zero_shot:
        print_zero_shot_results(avg_test_loss, avg_test_accuracy)
    else:
        print_evaluation_results(avg_test_loss, avg_test_accuracy)