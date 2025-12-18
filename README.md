# Overview of Project

## Motivation

Bank lenders face high volumes of loan approval requests every day and need a streamlined way to process them. We wanted to explore the possibility of using machine learning to automate loan approval request handling. To do so, we used a synthetic loan dataset and experimented with a few different models and techniques (K Nearest Neighbors, Logistic Regression, $l_1$ Regularization, etc.). 

## Insights from Exploratory Data Analysis

Our dataset can be foudn at the Kaggle badge below. Through our exploratory data analysis, we noticed some correlations between age, credit score, and debt with the likelihood of getting a loan approved. 

## Structure of Repository

- `data/`: Stores datasets used in the project
- `loan_tools/`: Contains the loan_tools package
- `LICENSE`: Legal terms under which the code and daa may be used
- `.gitignore`: Files that should not be tracked by Github
- `Makefile`: Provides convenient commands for running through the repository
- `.github/workflows` : Contains GitHub Actions workflows.  
- `main.ipynb` : Discusses the core parts of the analyses and results. 
- `environment.yml` : Specifies the environment and dependencies required to run the project.  
- `README.md` : Serves as the main entry point, describing the project purpose, structure, and usage.  
- `myst.yml` : Configures MyST build settings for the project.
- `eda_data_cleaning.ipynb`: Performs EDA and data cleaning on the dataset
- `logistic_regression.ipynb`: Fits logistic regression models to the dataset
- `knn.ipynb`: Fits k-nearest neighbors model to the dataset
- `references.bib`: Contains relevant information to references used in our analysis
- `pdf_builds/`: PDF versions of all notebooks
- `ai_documentation.txt`: Documents how AI tools were used in the making of the project.
- `plots/`: All relevant figures about the analysis. Note, the `plots/` folder will appear after the `eda_data_cleaning.ipynb` notebook is run via `make all`.
- `contribution_statement.md`: Contributions of each individual
- `project_description.md`: Description of the overall project
- `pyproject.toml`: File relevant for the installation of the loan_tools package


## Steps to Reproduce Our Findings

To reproduce our results locally, please follow the steps below:
1. In the terminal, run `make env`. This will configure the conda environment. Note that this will take some time to complete running.
2. Run `conda activate loanstatus` to activate the environment.
3. Run `pip install .` to install the dependencies specified in the `environment.yml`
4. Run to `make all` run all of the notebooks.
5. Check the plots folder, and ensure that the plots match up with what is displayed across the Jupyter Notebooks. To delete the plots, run `make clean`.
6. To test the `loan_tools` package, in the conda environment, run `python -m pytest`.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UCB-stat-159-f25/final-group11.git/HEAD?urlpath=%2Fdoc%2Ftree%2Feda_data_cleaning.ipynb)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17970326.svg)](https://doi.org/10.5281/zenodo.17971699)
[![Github Pages](https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white)](https://ucb-stat-159-f25.github.io/final-group11/)
[![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/parthpatel2130/realistic-loan-approval-dataset-us-and-canada)
