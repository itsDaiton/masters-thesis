import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.models_utils import get_last_layer
from utils.train_utils import print_training_results

    
def train(model, train_loader, val_loader, mode, config):
    model.to(config.device)
    
    if mode == 'linear_probing':
        for param in model.parameters():
            param.requires_grad = False
        for param in get_last_layer(model, model.model_name).parameters():
            param.requires_grad = True
    
    optimizer = config.optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    criterion = config.criterion
    
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_samples = 0
        
        for batch in tqdm(train_loader, desc='Train split'):
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
            for batch in tqdm(val_loader, desc='Validation split'):
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
    

def train_with_distillation(student, teacher, train_loader: DataLoader, val_loader: DataLoader, config):
    pass
        
def fine_tune_with_teacher(student, teacher, train_loader: DataLoader, val_loader: DataLoader, config):
    student.to(config.device)
    teacher.to(config.device)
        
    optimizer = config.optimizer(student.parameters(), lr=config.lr)
    criterion = config.criterion
    
    for epoch in range(config.num_epochs):
        student.train()
        teacher.eval()
        
        train_loss = 0
        train_correct = 0
        train_samples = 0
        
        for batch in tqdm(train_loader):
            images, labels = batch
            images, labels = images.to(config.device), labels.to(config.device)    
            optimizer.zero_grad()
            
            with torch.no_grad():
                teacher_outputs = teacher(images)
                _, teacher_predictions = torch.max(teacher_outputs, 1)
                
            student_outputs = student(images)
            loss = 0.5 * criterion(student_outputs, labels) + 0.5 * criterion(student_outputs, teacher_predictions)        
            
            loss.backward()
            optimizer.step()
                   
            train_loss += loss.item()
            
            _, predictions = torch.max(student_outputs, 1)
            
            train_correct += (predictions == labels).sum().item()
            train_samples += labels.size(0)
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_accuracy = train_correct / train_samples

        student.eval()
           
        val_loss = 0
        val_correct = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader):
                images, labels = batch
                images, labels = images.to(config.device), labels.to(config.device)
                
                teacher_outputs = teacher(images)
                _, teacher_predictions = torch.max(teacher_outputs, 1)
                students_outputs = student(images)
                loss = 0.5 * criterion(students_outputs, labels) + 0.5 * criterion(students_outputs, teacher_predictions)
                
                val_loss += loss.item()
                
                _, predictions = torch.max(students_outputs, 1)
                
                val_correct += (predictions == labels).sum().item()
                val_samples += labels.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = val_correct / val_samples
        
        print(f"""Epochs: {epoch + 1}/{config.num_epochs} 
              | train_loss: {avg_train_loss:.4f} 
              | train_acc: {avg_train_accuracy:.4f} 
              | val_loss: {avg_val_loss:.4f} 
              | val_acc: {avg_val_accuracy:.4f}""")   

def fine_tune_teacher(model, train_loader: DataLoader, val_loader: DataLoader, config):
    model.to(config.device)
    
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
            for batch in tqdm(val_loader):
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
        
        print(f"""Epoch: {epoch + 1}/{config.num_epochs} 
              | train_loss: {avg_train_loss:.4f} 
              | train_acc: {avg_train_accuracy:.4f} 
              | val_loss: {avg_val_loss:.4f} 
              | val_acc: {avg_val_accuracy:.4f}""")

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
            test_correct += (predictions == labels).sum().item()
            test_samples += labels.size(0)
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_accuracy = test_correct / test_samples

    print(f"test_loss: {avg_test_loss:.4f} | test_acc: {avg_test_accuracy:.4f}")