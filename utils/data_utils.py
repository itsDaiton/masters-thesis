from collections import defaultdict
import copy
import pandas as pd
from datasets import ClassLabel
from data.mappings.label_mappings import pcam, sun397


def clean_labels(dataset, name):
    """Custom label cleaning function specific for each dataset."""

    for split in dataset.keys():
        labels = dataset[split].features["label"].names
        labels_cleaned = []
        labels_cleaned = [s.replace("_", " ") for s in labels]
        labels_cleaned = [s.lower() for s in labels_cleaned]

        if name == "pcam":
            mappings = pcam
            labels_cleaned = [mappings.get(label, label) for label in labels]

        if name == "sun397":
            mappings = sun397
            labels_cleaned = [mappings.get(label, label) for label in labels]

        label_mapping = {i: labels_cleaned[i] for i in range(len(labels))}
        features_copy = dataset[split].features.copy()
        features_copy["label"] = ClassLabel(names=labels_cleaned)
        dataset[split] = dataset[split].cast(features_copy)
        dataset[split] = dataset[split].map(
            lambda example, label_mapping=label_mapping: {
                "label": label_mapping[example["label"]]
            }
        )


def create_few_shot_subset(dataset, k):
    dataset_subset = copy.deepcopy(dataset)
    label_to_indices = defaultdict(list)

    for idx in range(len(dataset_subset)):
        label = dataset_subset.get_label(idx)
        label_to_indices[label].append(idx)

    few_shot_indices = []
    for label, indices in label_to_indices.items():
        sample = indices[:k]
        few_shot_indices.extend(sample)

    dataset_subset.dataset = dataset_subset.dataset.select(few_shot_indices)

    return dataset_subset


def create_accuracy_dict(results, labels):
    accuracy_dict = dict(zip(labels, results))
    return accuracy_dict


def bold_string(string):
    return f"\033[1m{string}\033[0m"


def create_few_shot_table(data, n_shots, idx):
    df = pd.DataFrame(data, columns=[f"{n}-shot" for n in n_shots], index=idx)
    for i in range(len(n_shots) - 1):
        prev_idx = f"{n_shots[i]}-shot"
        next_idx = f"{n_shots[i + 1]}-shot"
        df[f"{next_idx} (% Δ)"] = ((df[next_idx] - df[prev_idx]) * 100).round(2)
    return df
