from torch.utils.data import Dataset

class ImageClassificationDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        processed_image = self.processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
        return processed_image, item['label']