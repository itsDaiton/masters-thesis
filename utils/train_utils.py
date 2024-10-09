import torch.nn as nn
   
def get_loss_function(is_binary_task=False):
    if is_binary_task:
        return nn.BCEWithLogitsLoss()
    else:
        return nn.CrossEntropyLoss() 

def print_training_results(epoch, total_epochs, train_loss, train_accuracy, val_loss, val_accuracy):
    print(f"Epochs: {epoch + 1}/{total_epochs} | train_loss: {train_loss:.4f} | train_acc: {train_accuracy:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_accuracy:.4f}")
    
def print_evaluation_results(test_loss, test_accuracy):
    print(f"test_loss: {test_loss:.4f} | test_acc: {test_accuracy:.4f}")
    
def print_zero_shot_results(loss, accuracy):
    print(f"Zero-shot evaluation completed: loss: {loss:.4f} | acc: {accuracy:.4f}")   

def calculate_hard_distillation(student_outputs, teacher_predictions, labels, criterion):
    return 0.5 * criterion(student_outputs, labels) + 0.5 * criterion(student_outputs, teacher_predictions)

def calculate_per_class_accuracy(labels, predictions):
    labels_unique = set(labels)
    per_class_accuracies = {}
    
    for label in labels_unique:
        label_indices = [i for i, l in enumerate(labels) if l == label]
        correct_predictions = sum([1 for i in label_indices if predictions[i] == label])
        per_class_accuracies[label] = correct_predictions / len(label_indices)
        
    return per_class_accuracies 