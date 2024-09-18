from torch.utils.data import Dataset
    
class ImageDataset(Dataset):
    """ Class for a custom image classification dataset, with the option to create captions for each image. """
    
    def __init__(self, dataset, processor, tokenizer=None, create_captions=False, prompt=None):
        self.dataset = dataset
        self.processor = processor
        self.tokenizer = tokenizer
        self.create_captions = create_captions
        self.prompt = prompt
        
        self.id2label = {i: label for i, label in enumerate(dataset.features['label'].names)}
        self.captions = self._create_captions_from_prompt() if self.create_captions and self.prompt and self.tokenizer else None
        self.tokenized_captions = self._tokenize_captions() if self.create_captions and self.prompt and self.tokenizer else None
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image, label = item['image'], item['label']    
        processed_image = self.processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
        
        return processed_image, label
    
    def get_labels(self):
        return self.dataset.features['label'].names 
    
    def get_captions(self):
        if not self.create_captions or not self.captions:
            raise ValueError("Captions were not created for this dataset. Set create_captions=True when creating the dataset.")      
        return self.captions
    
    def get_tokenized_captions(self):
        if not self.create_captions or not self.captions:
            raise ValueError("Captions were not created for this dataset. Set create_captions=True when creating the dataset.")
        return self.tokenized_captions        
         
    def get_label(self, idx):
        item = self.dataset[idx]
        label = item['label']
        
        return self.id2label[label]
            
    def _create_captions_from_prompt(self):
        if not self.prompt:
            raise ValueError("Prompt is not provided.")
        return [self.prompt.format(self.id2label[i]) for i in self.id2label]
           
    def _tokenize_captions(self):
        if not self.tokenizer:
            raise ValueError("Tokenizer is not provided.")
        return self.tokenizer(text=self.captions, return_tensors='pt', padding=True, truncation=True)
