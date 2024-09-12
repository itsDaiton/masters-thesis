def print_training_results(epoch, total_epochs, train_loss, train_accuracy, val_loss, val_accuracy):
    print(f"Epochs: {epoch + 1}/{total_epochs} 
          | train_loss: {train_loss:.4f} 
          | train_acc: {train_accuracy:.4f} 
          | val_loss: {val_loss:.4f} 
          | val_acc: {val_accuracy:.4f}")
    
def print_evaluation_results(test_loss, test_accuracy):
    print(f"test_loss: {test_loss:.4f} | test_acc: {test_accuracy:.4f}")

def calculate_hard_distillation(student_outputs, teacher_predictions, labels, criterion):
    return 0.5 * criterion(student_outputs, labels) + 0.5 * criterion(student_outputs, teacher_predictions)