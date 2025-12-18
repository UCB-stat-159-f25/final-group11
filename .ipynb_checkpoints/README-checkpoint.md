# Overview of Project

## Motivation

Bank lenders face high volumes of loan approval requests every day and need a streamlined way to process them. We wanted to explore the possibility of using machine learning to automate loan approval request handling. To do so, we used a synthetic loan dataset and experimented with a few different models and techniques (K Nearest Neighbors, Logistic Regression, L1 Regularization, etc.). 

## Insights from Exploratory Data Analysis

This is the link to our [dataset](https://www.kaggle.com/datasets/parthpatel2130/realistic-loan-approval-dataset-us-and-canada). Through our exploratory data analysis, we noticed some correlations between age, credit score, and debt with the likelihood of getting a loan approved. 

## Repository Structure

## Analysis + Conclusion

In our experimentation with machine learning models, we used a 99/1 train-test split since we had a large dataset (1,000,000 datapoints). The models we used had similar testing performance (~87%). Through our experimentation, we validated our original hypothesis that loan approvals/rejections followed some pattern, which we were able to accurately and consistently demonstrate with the machine learning models in our notebooks.

## Steps to Reproduce Our Findings

To reproduce our results locally, please follow the steps below:
1. In the terminal, run make. This will configure the conda environment. Note that this will take some time to complete running. 
2. Run each of the Jupyter Notebooks in sequential order
3. Check the plots folder, and ensure that the plots match up with what is displayed across the Jupyter Notebooks.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UCB-stat-159-f25/final-group11/2e4c50084b25f50349f7b791e438a18a8c192a5d?urlpath=lab%2Ftree%2FPart1.ipynb)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17970326.svg)](https://doi.org/10.5281/zenodo.17970326)

Here is a link to our [GitHub pages](https://ucb-stat-159-f25.github.io/final-group11/).
