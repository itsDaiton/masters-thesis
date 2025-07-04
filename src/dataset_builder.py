from torch.utils.data import Dataset
import seaborn as sns
import matplotlib.pyplot as plt


class ImageDataset(Dataset):
    def __init__(
        self,
        dataset,
        processor=None,
        tokenizer=None,
        create_captions=False,
        prompt=None,
    ):
        self.dataset = dataset
        self.processor = processor
        self.tokenizer = tokenizer
        self.create_captions = create_captions
        self.prompt = prompt

        self.id2label = {
            i: label for i, label in enumerate(dataset.features["label"].names)
        }
        self.captions = (
            self._create_captions_from_prompt()
            if self.create_captions and self.prompt and self.tokenizer
            else None
        )
        self.tokenized_captions = (
            self._tokenize_captions()
            if self.create_captions and self.prompt and self.tokenizer
            else None
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image, label = item["image"], item["label"]

        if image.mode == "L":
            image = image.convert("RGB")

        if self.processor is None:
            raise ValueError("Processor is not provided.")

        processed_image = self.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze()

        return processed_image, label

    def get_image(self, idx):
        return self.dataset[idx]["image"]

    def get_prompt(self):
        return self.prompt

    def get_labels(self):
        return self.dataset.features["label"].names

    def get_captions(self):
        if not self.create_captions or not self.captions:
            raise ValueError("Captions were turned off for this dataset.")
        return self.captions

    def get_tokenized_captions(self):
        if not self.create_captions or not self.captions:
            raise ValueError("Captions were turned off for this dataset.")
        return self.tokenized_captions

    def get_label(self, idx):
        item = self.dataset[idx]
        label = item["label"]

        return self.id2label[label]

    def get_num_classes(self):
        return len(self.id2label)

    def _create_captions_from_prompt(self):
        if not self.prompt:
            raise ValueError("Prompt is not provided.")
        return [
            self.prompt.format(*[self.id2label[i]] * self.prompt.count("{}"))
            for i in self.id2label
        ]

    def _tokenize_captions(self):
        if not self.tokenizer:
            raise ValueError("Tokenizer is not provided.")
        return self.tokenizer(
            text=self.captions, return_tensors="pt", padding=True, truncation=True
        )

    def set_prompt(self, new_prompt):
        self.prompt = new_prompt
        if self.create_captions:
            self.captions = self._create_captions_from_prompt()
            self.tokenized_captions = self._tokenize_captions()

    def set_processor(self, new_processor):
        self.processor = new_processor

    def plot_image(self, idx):
        sns.set_style("darkgrid")
        item = self.dataset[idx]
        image, label = item["image"], item["label"]
        label = self.id2label[label]

        plt.imshow(image.resize((224, 224)))
        plt.axis("off")
        plt.title(label)

        plt.grid(False)
        plt.tight_layout()
        plt.show()

    def augment_dataset(self, augmentations, batch_size=64):
        def augment_batch(batch):
            for i in range(len(batch["image"])):
                image = batch["image"][i]
                batch["image"][i] = augmentations(image)
            return batch

        self.dataset = self.dataset.map(
            augment_batch,
            batched=True,
            batch_size=batch_size,
            load_from_cache_file=False,
        )

        print(
            f"Augmentation completed. Total number of new samples generated: {len(self.dataset)}"
        )
