# Exploration and Comparison of Transformers for Image Classification

├── 📂 .github
│   └── 📂 workflows
│       └── 📄 pipeline.yml
├── 📄 .gitignore
├── 📄 .pylintrc
├── 📄 LICENSE
├── 📄 README.md
├── 📂 data
│   ├── 📄 __init__.py
│   ├── 📄 data_exploration.ipynb
│   └── 📂 mappings
│       └── 📄 label_mappings.py
├── 📂 experiments
│   ├── 📂 benchmarks
│   │   ├── 📂 few-shot-linear-probing
│   │   │   ├── 📄 few_shot_linear_probing_clip.ipynb
│   │   │   ├── 📄 few_shot_linear_probing_deit.ipynb
│   │   │   ├── 📄 few_shot_linear_probing_results.ipynb
│   │   │   ├── 📄 few_shot_linear_probing_swin.ipynb
│   │   │   └── 📄 few_shot_linear_probing_vit.ipynb
│   │   ├── 📂 fine-tuning-data-aug
│   │   │   ├── 📄 fine_tuning_data_aug_clip.ipynb
│   │   │   ├── 📄 fine_tuning_data_aug_deit.ipynb
│   │   │   ├── 📄 fine_tuning_data_aug_results.ipynb
│   │   │   ├── 📄 fine_tuning_data_aug_swin.ipynb
│   │   │   └── 📄 fine_tuning_data_aug_vit.ipynb
│   │   ├── 📂 fine-tuning
│   │   │   ├── 📄 fine_tuning_clip.ipynb
│   │   │   ├── 📄 fine_tuning_deit.ipynb
│   │   │   ├── 📄 fine_tuning_results.ipynb
│   │   │   ├── 📄 fine_tuning_swin.ipynb
│   │   │   └── 📄 fine_tuning_vit.ipynb
│   │   ├── 📂 linear-probing
│   │   │   ├── 📄 linear_probing_clip.ipynb
│   │   │   ├── 📄 linear_probing_deit.ipynb
│   │   │   ├── 📄 linear_probing_results.ipynb
│   │   │   ├── 📄 linear_probing_swin.ipynb
│   │   │   └── 📄 linear_probing_vit.ipynb
│   │   └── 📂 zero-shot-transfer
│   │       └── 📄 zero_shot_transfer.ipynb
│   ├── 📂 configs
│   │   ├── 📄 data_augmentations.json
│   │   ├── 📂 experiments
│   │   │   ├── 📄 few_shot_linear_probing.json
│   │   │   ├── 📄 fine_tuning.json
│   │   │   ├── 📄 fine_tuning_data_aug.json
│   │   │   ├── 📄 linear_probing.json
│   │   │   └── 📄 zero_shot_transfer.json
│   │   └── 📂 processors
│   │       ├── 📄 clip.json
│   │       ├── 📄 deit.json
│   │       ├── 📄 swin.json
│   │       └── 📄 vit.json
│   ├── 📄 experiments_results.ipynb
│   └── 📂 prerequisites
│       ├── 📄 install_packages.ipynb
│       ├── 📄 load_datasets.ipynb
│       └── 📄 load_models.ipynb
├── 📄 pyproject.toml
├── 📄 requirements.txt
├── 📂 src
│   ├── 📄 __init__.py
│   ├── 📄 dataset_builder.py
│   ├── 📄 models.py
│   └── 📄 train.py
└── 📂 utils
    ├── 📄 __init__.py
    ├── 📄 config.py
    ├── 📄 data_utils.py
    ├── 📄 models_utils.py
    ├── 📄 train_utils.py
    └── 📄 visualization_utils.py


## Description
TBD

## Built With
TBD

## Models
- CLIP
- ViT
- DeiT
- Swin

## Results
TBD

## Authors
- David Poslušný
