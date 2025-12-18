import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from loan_tools.fittools import get_classification_rate, best_alpha
import git
from pathlib import Path
import os

ROOT_DIR = Path(git.Repo('.', search_parent_directories=True).working_tree_dir)
DATA_PATH = os.path.join(ROOT_DIR, "data", "cleaned_data.csv")

sample_data = pd.read_csv(DATA_PATH)
sample_data = pd.get_dummies(sample_data, columns=['loan_intent', 'product_type', 'occupation_status'], drop_first=True)

@pytest.fixture
def fitted_model():
    y = sample_data['loan_status']
    X_cols = ['credit_score', 'loan_amount', 'annual_income', 'debt_to_income_ratio']
    X = sample_data[X_cols].astype(float)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    return model, X_cols


def test_get_classification_rate_output(fitted_model):
    model, X_cols = fitted_model
    
    result = get_classification_rate(model, X_cols, df=sample_data, y='loan_status')
    
    assert isinstance(result, tuple), "Should return a tuple"
    assert len(result) == 2, "Should return exactly 2 values"
    
    rate, cutoff = result
    
    assert isinstance(rate, (float, np.floating)), f"Rate should be float, got {type(rate)}"
    assert isinstance(cutoff, (float, np.floating)), f"Cutoff should be float, got {type(cutoff)}"
    
    assert 0 <= rate <= 100, f"Classification rate {rate} should be between 0 and 100"
    assert 0 <= cutoff <= 1, f"Cutoff {cutoff} should be between 0 and 1"
    
    cutoff_str = str(cutoff)
    if '.' in cutoff_str:
        decimal_places = len(cutoff_str.split('.')[1])
        assert decimal_places <= 2, f"Cutoff should have at most 2 decimal places, got {decimal_places}"


def test_best_alpha_selects_optimal_model():

    train_data = sample_data.iloc[:40000]
    test_data = sample_data.iloc[40000:]
    
    alphas = [0.01, 0.1, 1.0]
    
    best_model, best_classification, best_alpha_val = best_alpha(
        alphas=alphas,
        train=train_data,
        test=test_data,
        lp='l1',
        y_var='loan_status'
    )
    
    assert best_model is not None, "Best model should not be None"
    assert isinstance(best_model, LogisticRegression), "Should return LogisticRegression model"
    assert isinstance(best_classification, (tuple, list)), "Classification should be tuple or list"
    assert len(best_classification) == 2, "Classification should have 2 elements (rate, cutoff)"
    
    assert best_alpha_val in alphas, f"Best alpha {best_alpha_val} should be from {alphas}"
    
    rate, cutoff = best_classification
    assert 0 <= rate <= 100, f"Rate {rate} should be between 0 and 100"
    assert 0 <= cutoff <= 1, f"Cutoff {cutoff} should be between 0 and 1"
    
    expected_C = 1 / best_alpha_val
    
    assert np.isclose(best_model.C, expected_C), f"Model C={best_model.C} should match 1/alpha={expected_C}"
    assert rate > 50, f"Classification rate {rate}% should be better than random guessing"