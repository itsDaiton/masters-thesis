import torch
from tqdm import tqdm
from utils.models_utils import get_last_layer
from utils.train_utils import print_training_results, print_evaluation_results, calculate_hard_distillation

    
def train(model, train_loader, val_loader, config, model_type, mode):
    model.to(config.device)
    
    if mode == 'linear_probing':
        for param in model.parameters():
            param.requires_grad = False
        for param in get_last_layer(model, model_type).parameters():
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
            for batch in tqdm(val_loader, desc='Val'):
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
        
        print_training_results(epoch, config.num_epochs, avg_train_loss, avg_train_accuracy, avg_val_loss, avg_val_accuracy)
    
def train_with_distillation(student, teacher, train_loader, val_loader, config, model_type, mode):
    student.to(config.device)
    teacher.to(config.device)
    
    if mode == 'linear_probing':
        for param in student.parameters():
            param.requires_grad = False
        for param in get_last_layer(student, model_type).parameters():
            param.requires_grad = True
    else:
        for param in student.parameters():
            param.requires_grad = True
            
    optimizer = config.optimizer(filter(lambda p: p.requires_grad, student.parameters()), lr=config.lr)
    criterion = config.criterion
    
    for epoch in range(config.num_epochs):
        student.train()
        teacher.eval()
        
        train_loss = 0
        train_correct = 0
        train_samples = 0
        
        for batch in tqdm(train_loader, desc='Train'): 
            images, labels = batch
            images, labels = images.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                teacher_outputs = teacher(images)
                _, teacher_predictions = torch.max(teacher_outputs, 1)
                
            student_outputs = student(images)
            _, predictions = torch.max(student_outputs, 1)
             
            loss = calculate_hard_distillation(student_outputs, teacher_predictions, labels, criterion)    
            loss.backward()
            optimizer.step()
                   
            train_loss += loss.item()               
            train_correct += (predictions == labels).sum().item()
            train_samples += labels.size(0)
                  
        avg_train_loss = train_loss / len(train_loader)
        avg_train_accuracy = train_correct / train_samples
        
        student.eval()
        
        val_loss = 0
        val_correct = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Val'):
                images, labels = batch
                images, labels = images.to(config.device), labels.to(config.device)
                
                teacher_outputs = teacher(images)
                _, teacher_predictions = torch.max(teacher_outputs, 1)
                students_outputs = student(images)
                _, predictions = torch.max(students_outputs, 1)
                
                loss = calculate_hard_distillation(students_outputs, teacher_predictions, labels, criterion)
           
                val_loss += loss.item()
                val_correct += (predictions == labels).sum().item()
                val_samples += labels.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = val_correct / val_samples
        
        print_training_results(epoch, config.num_epochs, avg_train_loss, avg_train_accuracy, avg_val_loss, avg_val_accuracy)

def evaluate(model, test_loader, config):  
    model.to(config.device)
    model.eval()

    criterion = config.criterion
    
    test_loss = 0
    test_correct = 0
    test_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Test'):
            images, labels = batch
            images, labels = images.to(config.device), labels.to(config.device)
            
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            test_correct += (predictions == labels).sum().item()
            test_samples += labels.size(0)
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_accuracy = test_correct / test_samples

    print_evaluation_results(avg_test_loss, avg_test_accuracy)