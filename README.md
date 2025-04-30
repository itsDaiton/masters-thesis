# Exploration and Comparison of Transformers for Image Classification
This repostiry contains the complete source code for the  master thesis ***Exploration and Comparison of Transformers for Image Classification*** by David Poslušný at [***Faculty of Informatics and Statistics of Prague University of Economics and Business***](https://fis.vse.cz).


This thesis focuses on the research and application of Transformers in image classification. The main goal is to evaluate the performance of selected Transformer models on diverse image classification datasets in a series of hand-crafted transfer learning experiments. The results of the analysis are compared and further discussed in the context of available literature. The first part of the thesis presents a theoretical background of Transformers. It contains an overview of the original architecture and its adaptation to computer vision (CV). An extensive literature survey of Transformers in CV is performed, focusing primarily on image classification. The survey provides a basis for selecting models used in the experiments. The second part reviews available image classification datasets to select diverse domain-specific challenges for the models. Subsequently, experiments based on transfer learning are designed to evaluate and compare the models in various experimental setups. The thesis concludes by applying the selected models to the unseen data. The results are evaluated, analyzed, and discussed based on the context of the individual experiments. The primary contribution of the thesis is an in-depth evaluation of key Transformer models adopted for image classification on diverse domain-specific datasets. The complex experiments analyze the models under different conditions and compare their generalization performance outside of the training distribution. The outcome serves as a summary of the most important Transformer-based architectures for image classification.

## Table of Contents
  * [Models](#models)
  * [Datasets](#datasets)
  * [Implementation](#implementation)
  * [Experiments](#experiments)
    + [Linear probing](#linear-probing)
    + [Fine-tuning](#fine-tuning)
    + [Few-shot linear probing](#few-shot-linear-probing)
    + [Fine-tuning with data augmentations](#fine-tuning-with-data-augmentations)
    + [Zero-shot transfer](#zero-shot-transfer)
  * [Configurations](#configurations)
  * [Prerequisites](#prerequisites)
  * [Results](#results)
  * [Requirements](#requirements)
  * [How to Run the Project](#how-to-run-the-project)
  * [License](#license)
  * [Acknowledgements](#acknowledgements)
  * [Authors](#authors)


## Models
- **Vision Transformer (ViT)**
    - https://huggingface.co/google/vit-base-patch16-224
- **Data-efficient image Transformer (DeiT)**
    - https://huggingface.co/facebook/deit-base-distilled-patch16-224
- **Swin Transformer (Swin)**
    - https://huggingface.co/microsoft/swin-base-patch4-window7-224
- **Contrastive Language-Image Pre-training (CLIP)**
    - https://huggingface.co/openai/clip-vit-base-patch16

## Datasets
- **RESISC45**
    - https://huggingface.co/datasets/timm/resisc45
- **Food-101**
    - https://huggingface.co/datasets/ethz/food101
- **FER2013**
    - https://huggingface.co/datasets/AutumnQiu/fer2013
- **PatchCamelyon**
    - https://huggingface.co/datasets/zacharielegault/PatchCamelyon 
- **SUN397**
    - https://huggingface.co/datasets/dpdl-benchmark/sun397 
- **DTD**
    - https://huggingface.co/datasets/tanganke/dtd
 

## Implementation
The main logic behind the implementation in the exeperiments is located in:

```
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
 
## Experiments

### Linear probing
```
├── 📂 experiments
│   ├── 📂 benchmarks
│   │   ├── 📂 linear-probing
│   │   │   ├── 📄 linear_probing_clip.ipynb
│   │   │   ├── 📄 linear_probing_deit.ipynb
│   │   │   ├── 📄 linear_probing_results.ipynb
│   │   │   ├── 📄 linear_probing_swin.ipynb
│   │   │   └── 📄 linear_probing_vit.ipynb
```
### Fine-tuning
```
├── 📂 experiments
│   ├── 📂 benchmarks
│   │   ├── 📂 fine-tuning
│   │   │   ├── 📄 fine_tuning_clip.ipynb
│   │   │   ├── 📄 fine_tuning_deit.ipynb
│   │   │   ├── 📄 fine_tuning_results.ipynb
│   │   │   ├── 📄 fine_tuning_swin.ipynb
│   │   │   └── 📄 fine_tuning_vit.ipynb
```
### Few-shot linear probing
```
├── 📂 experiments
│   ├── 📂 benchmarks
│   │   ├── 📂 few-shot-linear-probing
│   │   │   ├── 📄 few_shot_linear_probing_clip.ipynb
│   │   │   ├── 📄 few_shot_linear_probing_deit.ipynb
│   │   │   ├── 📄 few_shot_linear_probing_results.ipynb
│   │   │   ├── 📄 few_shot_linear_probing_swin.ipynb
│   │   │   └── 📄 few_shot_linear_probing_vit.ipynb
```
### Fine-tuning with data augmentations
```
├── 📂 experiments
│   ├── 📂 benchmarks
│   │   ├── 📂 fine-tuning-data-aug
│   │   │   ├── 📄 fine_tuning_data_aug_clip.ipynb
│   │   │   ├── 📄 fine_tuning_data_aug_deit.ipynb
│   │   │   ├── 📄 fine_tuning_data_aug_results.ipynb
│   │   │   ├── 📄 fine_tuning_data_aug_swin.ipynb
│   │   │   └── 📄 fine_tuning_data_aug_vit.ipynb
```
### Zero-shot transfer
```
├── 📂 experiments
│   ├── 📂 benchmarks
│   │   └── 📂 zero-shot-transfer
│   │       └── 📄 zero_shot_transfer.ipynb
```

## Configurations
All experiment and model settings are stored in:
```
├── 📂 experiments
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
```

## Prerequisites
`install_packages.ipynb` 
- install all necessary packages

`load_datasets.ipynb` 
- pre-load all the available datasets

`load_models.ipynb` 
- pre-load all the available models and their processors

```
├── 📂 experiments
│   └── 📂 prerequisites
│       ├── 📄 install_packages.ipynb 
│       ├── 📄 load_datasets.ipynb
│       └── 📄 load_models.ipynb
```

## Results
The complete results with additional analyses are avaialble in Jupyter nootebok file located at:
```
├── 📂 experiments
│   ├── 📄 experiments_results.ipynb
```

For the results of individual experiments, please refer to specific Jupyter notebooks in `experiments/benchmarks/` folder.

## Requirements
To run this project, ensure the following requirements are installed:
- Python 3.10 or higher
- Dependencies listed in `requirements.txt`

Install the dependencies using this command:
```
pip install -r requirements.txt
```

## How to Run the Project
1. Clone the repository:
    ```
    git clone https://github.com/itsDaiton/masters-thesis.git
    cd masters-thesis
    ```
2. Install the prerequisites for the experiments:
    ```
    cd experiments/prerequisites
    jupyter notebook install_packages.ipynb 
    jupyter notebook load_datasets.ipynb
    jupyter notebook load_models.ipynb
    ```
3.  Run an experiment, e.g.:
    ```
    cd experiments/benchmarks/fine-tuning
    jupyter notebook fine_tuning_vit.ipynb
    ```

## License
The project is distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements
Thesis supervisor:  doc. Ing. Ondřej Zamazal, Ph.D.

This work was also supported by the Ministry of Education, Youth and Sports of the Czech
Republic through the e-INFRA CZ (ID:90254).

## Authors
Bc. David Poslušný <br>
Study program: Knowledge and Web Technologies <br>
Specialization: Quantitative Analysis <br>
[Prague University of Economics and Business Faculty of Informatics and Statistics](https://fis.vse.cz/)
