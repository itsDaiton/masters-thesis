import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from utils.models_utils import get_last_layer
from utils.data_utils import create_few_shot_subset
from utils.train_utils import (
    print_training_results,
    print_evaluation_results,
    print_zero_shot_results,
    calculate_hard_distillation,
    calculate_per_class_accuracy,
    EarlyStopping,
)


def train_model(
    model,
    train,
    config,
    architecture,
    val=None,
    use_val=True,
    fine_tune=True,
    with_distillation=False,
    teacher=None,
    few_shot=None,
    use_early_stopping=False,
    gradient_clipping=False,
    scheduling=False,
):
    if few_shot is not None:
        train = create_few_shot_subset(train, few_shot)

    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
    if use_val:
        val_loader = DataLoader(val, batch_size=config.batch_size)

    if use_early_stopping:
        if not use_val:
            raise ValueError(
                "Early stopping can only be used if a validation set is provided."
            )
        early_stopping = EarlyStopping(
            patience=config.early_stopping.patience, delta=config.early_stopping.delta
        )

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

    optimizer = config.optimizer(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = config.criterion

    if scheduling:
        warmup_scheduler = LinearLR(
            optimizer=optimizer,
            start_factor=0.1,
            end_factor=1,
            total_iters=config.scheduler.warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config.num_epochs - config.scheduler.warmup_epochs,
            eta_min=config.scheduler.eta_min,
        )
        scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.scheduler.warmup_epochs],
        )

    for epoch in range(config.num_epochs):
        model.train()

        train_loss = 0
        train_correct = 0
        train_samples = 0

        total_train_labels = []
        total_train_predictions = []

        for batch in tqdm(train_loader):
            images, labels = batch
            images, labels = images.to(config.device), labels.to(config.device)
            optimizer.zero_grad()

            if with_distillation and teacher is not None:
                with torch.no_grad():
                    teacher_outputs = teacher(images)
                    _, teacher_predictions = torch.max(teacher_outputs, 1)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            if with_distillation and teacher is not None:
                loss = calculate_hard_distillation(
                    outputs, teacher_predictions, labels, criterion
                )
            else:
                loss = criterion(outputs, labels)

            loss.backward()

            if gradient_clipping:
                clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=config.gradient_clipping.max_norm,
                )

            optimizer.step()

            train_loss += loss.item()
            train_correct += (predictions == labels).sum().item()
            train_samples += labels.size(0)

            total_train_labels.extend(labels.cpu().numpy())
            total_train_predictions.extend(predictions.cpu().numpy())

        if scheduling:
            scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_accuracy = train_correct / train_samples

        train_per_class_accuracies = calculate_per_class_accuracy(
            total_train_labels,
            total_train_predictions,
            train.get_labels(),
        )

        if use_val:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_samples = 0

            val_total_labels = []
            val_total_predictions = []

            with torch.no_grad():
                for batch in tqdm(val_loader):
                    images, labels = batch
                    images, labels = images.to(config.device), labels.to(config.device)

                    if with_distillation and teacher is not None:
                        teacher_outputs = teacher(images)
                        _, teacher_predictions = torch.max(teacher_outputs, 1)

                    outputs = model(images)
                    _, predictions = torch.max(outputs, 1)

                    if with_distillation and teacher is not None:
                        loss = calculate_hard_distillation(
                            outputs, teacher_predictions, labels, criterion
                        )
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

                if use_early_stopping:
                    early_stopping(avg_val_loss, model)
                    if early_stopping.early_stop:
                        print(
                            "Early stopping triggered. Stopping training and saving the model..."
                        )
                        early_stopping.load_best_model(model)
                        break

        if use_val:
            print_training_results(
                epoch,
                config.num_epochs,
                avg_train_loss,
                avg_train_accuracy,
                avg_val_loss,
                avg_val_accuracy,
            )
        else:
            print_training_results(
                epoch, config.num_epochs, avg_train_loss, avg_train_accuracy
            )

    if use_val:
        return (
            avg_train_loss,
            avg_train_accuracy,
            train_per_class_accuracies,
            avg_val_loss,
            avg_val_accuracy,
            val_per_class_accuracies,
        )
    return (
        avg_train_loss,
        avg_train_accuracy,
        train_per_class_accuracies,
    )


def evaluate_model(model, data, config, zero_shot=False):
    if zero_shot:
        tokenized_captions = data.get_tokenized_captions()
        input_ids = tokenized_captions["input_ids"].to(config.device)

    dataloader = DataLoader(data, batch_size=config.batch_size)
    criterion = config.criterion

    model.to(config.device)
    model.eval()

    test_loss = 0
    test_correct = 0
    test_samples = 0

    total_labels = []
    total_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
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

    return (avg_test_loss, avg_test_accuracy, per_class_accuracies)


def zero_shot_predict(model, image, processor, tokenizer, captions, config):
    model.to(config.device)
    images = processor(images=image, return_tensors="pt")["pixel_values"]
    input_ids = tokenizer(
        text=captions, return_tensors="pt", padding=True, truncation=True
    )["input_ids"]
    images, input_ids = images.to(config.device), input_ids.to(config.device)
    outputs = model(images=images, texts=input_ids)
    probs = outputs.softmax(dim=1)
    return probs
