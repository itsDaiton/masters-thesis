from datasets import ClassLabel
from data.mappings.label_mappings import pcam, sun397
from collections import defaultdict
import copy


def clean_labels(dataset, name):
    """ Custom label cleaning function specific for each dataset.""" 
    
    for split in dataset.keys():
        labels = dataset[split].features['label'].names
        clean_labels = []
        clean_labels = [s.replace('_', ' ') for s in labels]
        clean_labels = [s.lower() for s in clean_labels]
        
        if name == 'pcam':
            mappings = pcam
            clean_labels = [mappings.get(label, label) for label in labels]
            
        if name == 'sun397':
            mappings = sun397
            clean_labels = [mappings.get(label, label) for label in labels]
        
        label_mapping = {i: clean_labels[i] for i in range(len(labels))}
        features_copy = dataset[split].features.copy()
        features_copy['label'] = ClassLabel(names=clean_labels)
        dataset[split] = dataset[split].cast(features_copy) 
        dataset[split] = dataset[split].map(lambda example: {'label' : label_mapping[example['label']]})   

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
