import torch
from torchvision import transforms


class EarlyStopping:
    def __init__(self, patience=3, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.best_state = None
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        else:
            raise ValueError("No best model found.")


def get_gpu_info():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"VRAM: {vram:.2f} GB")
    else:
        print("CUDA is not available.")


def print_training_results(
    epoch, total_epochs, train_loss, train_accuracy, val_loss=None, val_accuracy=None
):
    if val_loss is not None and val_accuracy is not None:
        print(
            f"Epochs: {epoch + 1}/{total_epochs} | train_loss: {train_loss:.4f} | "
            f"train_acc: {train_accuracy:.4f} | val_loss: {val_loss:.4f} | "
            f"val_acc: {val_accuracy:.4f}"
        )
    else:
        print(
            f"Epochs: {epoch + 1}/{total_epochs} | train_loss: {train_loss:.4f} | "
            f"train_acc: {train_accuracy:.4f}"
        )


def print_evaluation_results(test_loss, test_accuracy):
    print(f"test_loss: {test_loss:.4f} | test_acc: {test_accuracy:.4f}")


def print_zero_shot_results(loss, accuracy):
    print(f"Zero-shot evaluation completed: loss: {loss:.4f} | acc: {accuracy:.4f}")


def calculate_hard_distillation(
    student_outputs, teacher_predictions, labels, criterion
):
    return 0.5 * criterion(student_outputs, labels) + 0.5 * criterion(
        student_outputs, teacher_predictions
    )


def calculate_per_class_accuracy(labels, predictions, class_names):
    labels_unique = set(labels)
    per_class_accuracies = {}

    for label in labels_unique:
        label_indices = [i for i, l in enumerate(labels) if l == label]
        correct_predictions = sum([1 for i in label_indices if predictions[i] == label])
        per_class_accuracies[class_names[label]] = correct_predictions / len(
            label_indices
        )

    return per_class_accuracies


def get_top_5_accuracies(per_class_accuracies):
    sorted_accuracies = sorted(
        per_class_accuracies.items(), key=lambda item: item[1], reverse=True
    )
    return sorted_accuracies[:5]


def get_bottom_5_accuracies(per_class_accuracies):
    sorted_accuracies = sorted(per_class_accuracies.items(), key=lambda item: item[1])
    return sorted_accuracies[:5]


def get_data_augmentations():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomInvert(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3
            ),
        ]
    )
