import torch
from tqdm import tqdm
from utils.models_utils import get_last_layer
from torch.utils.data import DataLoader
from utils.train_utils import (
    print_training_results,
    print_evaluation_results,
    print_zero_shot_results,
    calculate_hard_distillation,
    calculate_per_class_accuracy,
    get_loss_function,
)
    
def train_model(model, train, val, config, architecture, fine_tune=True, with_distillation=False, teacher=None):
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=config.batch_size)
    
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
    criterion = get_loss_function(config.is_binary_task)
    
    for epoch in range(config.num_epochs):
        model.train()
        
        train_loss = 0
        train_correct = 0
        train_samples = 0
        
        total_train_labels = []
        total_train_predictions = []
        
        for batch in tqdm(train_loader, desc='Train'):
            images, labels = batch
            images, labels = images.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            
            if with_distillation and teacher is not None:
                with torch.no_grad():
                    teacher_outputs = teacher(images)
                    _, teacher_predictions = torch.max(teacher_outputs, 1)  
                    
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            
            if config.is_binary_task:
                outputs = outputs[:, 1]
                predictions = (torch.sigmoid(outputs) > 0.5).long()
                labels = labels.float()
            
            if with_distillation and teacher is not None:
                loss = calculate_hard_distillation(outputs, teacher_predictions, labels, criterion)
            else:
                loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (predictions == labels).sum().item()
            train_samples += labels.size(0)
            
            total_train_labels.extend(labels.cpu().numpy())
            total_train_predictions.extend(predictions.cpu().numpy())
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_accuracy = train_correct / train_samples
        
        train_per_class_accuracies = calculate_per_class_accuracy(
            total_train_labels, 
            total_train_predictions,
            train.get_labels(),
        )
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0
        
        val_total_labels = []
        val_total_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Val'):
                images, labels = batch
                images, labels = images.to(config.device), labels.to(config.device)
                
                if with_distillation and teacher is not None:
                    teacher_outputs = teacher(images)
                    _, teacher_predictions = torch.max(teacher_outputs, 1)
                    
                outputs = model(images)
                _, predictions = torch.max(outputs, 1)
                
                if config.is_binary_task:
                    outputs = outputs[:, 1]
                    predictions = (torch.sigmoid(outputs) > 0.5).long()
                    labels = labels.float()
                
                if with_distillation and teacher is not None:
                    loss = calculate_hard_distillation(outputs, teacher_predictions, labels, criterion)
                else:
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item()
                val_correct += (predictions == labels).sum().item()
                val_samples += labels.size(0)
                
                val_total_labels.extend(labels.cpu().numpy())
                val_total_predictions.extend(predictions.cpu().numpy())
                                      
            avg_val_loss = val_loss / len(val_loader)
            avg_val_accuracy = val_correct / val_samples
            
            val_per_class_accuracies = calculate_per_class_accuracy(
                val_total_labels, 
                val_total_predictions,
                val.get_labels(),
            )
            
            print_training_results(epoch, config.num_epochs, avg_train_loss, avg_train_accuracy, avg_val_loss, avg_val_accuracy)
            
            return (
                avg_train_loss,
                avg_train_accuracy,
                train_per_class_accuracies,
                avg_val_loss,
                avg_val_accuracy,
                val_per_class_accuracies,
            ) 

def evaluate_model(model, data, config, zero_shot=False):
    if zero_shot:
        tokenized_captions = data.get_tokenized_captions()
        input_ids = tokenized_captions['input_ids'].to(config.device)
        
    dataloader = DataLoader(data, batch_size=config.batch_size) 
    criterion = get_loss_function(config.is_binary_task)
    
    model.to(config.device)
    model.eval()
    
    test_loss = 0
    test_correct = 0
    test_samples = 0
    
    total_labels = []
    total_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Test' if not zero_shot else 'Zero-shot'):
            images, labels = batch
            images, labels = images.to(config.device), labels.to(config.device)
            
            if zero_shot:
                outputs = model(images=images, texts=input_ids)
            else:
                outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            
            if config.is_binary_task:
                outputs = outputs[:, 1]
                predictions = (torch.sigmoid(outputs) > 0.5).long()
                labels = labels.float()
            
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            test_correct += (predictions == labels).sum().item()
            test_samples += labels.size(0)
            
            total_labels.extend(labels.cpu().numpy())
            total_predictions.extend(predictions.cpu().numpy())
    
    avg_test_loss = test_loss / len(dataloader)
    avg_test_accuracy = test_correct / test_samples
    
    per_class_accuracies = calculate_per_class_accuracy(
        total_labels, 
        total_predictions,
        data.get_labels(),
    )
    
    if zero_shot:
        print_zero_shot_results(avg_test_loss, avg_test_accuracy)
    else:
        print_evaluation_results(avg_test_loss, avg_test_accuracy)
    
    return (
        avg_test_loss,
        avg_test_accuracy, 
        per_class_accuracies
    )
        
def zero_shot_predict(model, image, processor, tokenizer, captions):
    images = processor(images=image, return_tensors='pt')['pixel_values']
    input_ids = tokenizer(text=captions, return_tensors='pt', padding=True, truncation=True)['input_ids']
    outputs = model(images=images, texts=input_ids)  
    probs = outputs.softmax(dim=1)
    return probs