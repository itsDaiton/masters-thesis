from datasets import load_dataset, ClassLabel
from data.country_codes import get_country_codes

data = load_dataset('timm/resisc45')
data = load_dataset('dpdl-benchmark/sun397', split=['va'])

def clean_labels(dataset, name):
    """ Custom label cleaning function specific for each dataset.""" 
    
    for split in dataset.keys():
        labels = dataset[split].features['label'].names
        clean_labels = []
        clean_labels = [s.replace('_', ' ') for s in labels]
        clean_labels = [s.lower() for s in clean_labels]
        
        if name == 'pcam':
            clean_labels = ['healthy lymph node tissue', 'lymph node tumor tissue'] 
            
        if name == 'country211':
            mappings = get_country_codes()
            clean_labels = [mappings.get(label, label) for label in labels]   
        
        label_mapping = {i: clean_labels[i] for i in range(len(labels))}
        features_copy = dataset[split].features.copy()
        features_copy['label'] = ClassLabel(names=clean_labels)
        dataset[split] = dataset[split].cast(features_copy) 
        dataset[split] = dataset[split].map(lambda example: {'label' : label_mapping[example['label']]})