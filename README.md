# Exploration and Comparison of Transformers for Image Classification
This repostiry contains the complete source code for the  master thesis ***Exploration and Comparison of Transformers for Image Classification*** by David Poslušný at [***Faculty of Informatics and Statistics of Prague University of Economics and Business***](https://fis.vse.cz).

This thesis focuses on the research and application of Transformers in image classification. The main goal is to evaluate the performance of selected Transformer models on diverse image classification datasets in a series of hand-crafted transfer learning experiments. The results of the analysis are compared and further discussed in the context of available literature.

The first part of the thesis presents a theoretical background of Transformers. It contains an overview of the original architecture and its adaptation to computer vision (CV). An extensive literature survey of Transformers in CV is performed, focusing primarily on image classification. The survey provides a basis for selecting models used in the experiments. The second part reviews available image classification datasets to select diverse domain-specific challenges for the models. Subsequently, experiments based on transfer learning are designed to evaluate and compare the models in various experimental setups. The thesis concludes by applying the selected models to the unseen data. The results are evaluated, analyzed, and discussed based on the context of the individual experiments.

The primary contribution of the thesis is an in-depth evaluation of key Transformer models adopted for image classification on diverse domain-specific datasets. The complex experiments analyze the models under different conditions and compare their generalization performance outside of the training distribution. The outcome serves as a summary of the most important Transformer-based architectures for image classification.

## Repository Structure
```
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
```

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
