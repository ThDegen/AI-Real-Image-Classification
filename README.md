## Project Description

Name: AI vs. Authentic Image Classification
The goal of our project is to develop a classification system able to identify AI-generated images (synthetic) vs. authentic photography (real).

Dataset (https://www.kaggle.com/competitions/detect-ai-vs-human-generated-images):
The dataset consists of authentic images from the Shutterstock platform, categorised into different groups. These categories include one-third of images featuring humans, with the remaining images featuring a balanced selection of other categories. Each authentic image was paired with an AI-generated image. The AI images were created using state-of-the-art generative models, enabling direct comparison between real and synthetic content. Labels for the training and test data are provided in train.csv and test.csv, Binary (0 = Real, 1 = AI-generated). The training set includes 79,950 images, while the test set includes 19,986.

Model:
To separate AI-generated images from real ones, we aim to evaluate the performance of the ResNet model family pretrained on ImageNet.

## Machine Learning Operations Canvas

[Read the canvas](reports/MLOps_Canvas_Group_75.pdf)

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
