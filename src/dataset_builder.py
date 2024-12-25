import random
from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, concatenate_datasets
import seaborn as sns
import matplotlib.pyplot as plt


class ImageDataset(Dataset):
    """
    Class for a custom image classification dataset buidling.
    The class has an option to create captions for each image (CLIP).
    """

    def __init__(
        self,
        dataset,
        processor,
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

    def augment_dataset(self, augmentations, percentage=0.5):
        if not 0 <= percentage <= 1:
            raise ValueError("Percentage must be between 0 and 1")

        num_samples = int(len(self.dataset) * percentage)
        indices = random.sample(range(len(self.dataset)), num_samples)

        augmented_data = {key: [] for key in self.dataset.features}

        with tqdm(
            total=num_samples,
            desc="Generating new samples...",
            dynamic_ncols=True,
            leave=False,
        ) as pbar:
            for idx in indices:
                item = self.dataset[idx]
                image, label = item["image"], item["label"]

                if image.mode == "L":
                    image = image.convert("RGB")

                augmented_image = augmentations(image)

                augmented_data["image"].append(augmented_image)
                augmented_data["label"].append(label)

                if "image_id" in self.dataset.features:
                    image_id = item.get("image_id", f"aug_{idx}")
                    augmented_data["image_id"].append(image_id)

                pbar.update(1)

            tqdm.write("Creating augmented dataset...")
            features = self.dataset.features
            augmented_dataset = HFDataset.from_dict(augmented_data, features=features)

            with tqdm(
                total=1,
                desc="Concatenating new samples...",
                dynamic_ncols=True,
                leave=True,
            ) as pbar:
                self.dataset = concatenate_datasets([self.dataset, augmented_dataset])
                pbar.update(1)

            tqdm.write(
                f"Augmentation completed. Total number of new images generated: {num_samples}."
            )
