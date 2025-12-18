import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm
from loan_tools.fittools import get_classification_rate, best_alpha
import git
from pathlib import Path
import os

ROOT_DIR = Path(git.Repo('.', search_parent_directories=True).working_tree_dir)
DATA_PATH = os.path.join(ROOT_DIR, "data", "cleaned_data.csv")

# Load actual data instead of generating random data
sample_data = pd.read_csv(DATA_PATH)
sample_data = pd.get_dummies(sample_data, columns = ['loan_intent', 'product_type', 'occupation_status'], drop_first = True)

def test_classification_rate():
    y = sample_data['loan_status']
    X_cols_naive = ['loan_intent_Debt Consolidation', 'loan_intent_Education',
           'loan_intent_Home Improvement', 'loan_intent_Medical',
           'loan_intent_Personal', 'product_type_Line of Credit',
           'product_type_Personal Loan', 'credit_score']
    X_naive = sm.add_constant(sample_data[X_cols_naive].astype(int))
    
    model_naive = sm.Logit(y, X_naive).fit()
    class_rate = get_classification_rate(md = model_naive, X = X_cols_naive, df = test)
    assert type(class_rate) == tuple
    assert type(class_rate[0]) == float