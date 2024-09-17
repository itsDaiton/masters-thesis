from torch.utils.data import Dataset
    
class ImageDataset(Dataset):
    def __init__(self, dataset, processor, tokenizer=None, prompt=None, create_captions=False):
        self.dataset = dataset
        self.processor = processor
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.create_captions = create_captions
        
        self.id2label = {i: label for i, label in enumerate(dataset.features['label'].names)}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image, label = item['image'], item['label']
        
        processed_image = self.processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
        
        if self.create_captions and self.tokenizer and self.prompt:
            caption = self._create_caption_from_prompt(label)
            processed_text = self.tokenizer(text=caption, return_tensors='pt', padding=True, truncation=True,)['input_ids']
            return processed_image, processed_text, label
        
        return processed_image, label
            
    def _create_caption_from_prompt(self, label):
        class_label = self.id2label[label]
        
        return self.prompt.format(class_label)

    def get_caption(self, idx):
        
        if not self.create_captions:
            raise ValueError("Captions were not created for this dataset. Set make_captions=True when creating the dataset.")
        
        item = self.dataset[idx]
        label = item['label']
        
        return self._create_caption_from_prompt(label)
    
    def get_label(self, idx):
        item = self.dataset[idx]
        label = item['label']
        return self.id2label[label]