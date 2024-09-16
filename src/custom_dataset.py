from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
        self.num_classes = self.dataset.features['label'].num_classes
        
        self.id2label = {i: label for i, label in enumerate(dataset.features['label'].names)}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        processed_image = self.processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
        return processed_image, item['label']
    
    def get_label(self, idx):
        item = self.dataset[idx]
        label = item['label']
        return self.id2label[label]
    
class ImageDatasetWithCaptions(Dataset):
    def __init__(self, dataset, processor, tokenizer, prompt):
        self.dataset = dataset
        self.processor = processor
        self.tokenizer = tokenizer
        self.prompt = prompt
        
        self.id2label = {i: label for i, label in enumerate(dataset.features['label'].names)}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image, label = item['image'], item['label']
        processed_image = self.processor(images=image, return_tensors='pt')['pixel_values'].squeeze()    
        caption = self.create_caption_from_prompt(label)   
        processed_text = self.tokenizer(text=caption, return_tensors='pt', padding=True, truncation=True,)['input_ids']
        
        return processed_image, processed_text, label  
    
    def create_caption_from_prompt(self, label):
        class_label = self.dataset.features['label'].int2str(label)
        
        return self.prompt.format(class_label)
    
    def get_caption(self, idx):
        item = self.dataset[idx]
        label = item['label']
        
        return self.create_caption_from_prompt(label)
    
    def get_label(self, idx):
        item = self.dataset[idx]
        label = item['label']
        return self.id2label[label]