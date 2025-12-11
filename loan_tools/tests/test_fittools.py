import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm
from fittools import get_classification_rate, best_alpha 

sample_data = pd.DataFrame({
    'loan_status': [0, 1, 0, 1, 0, 1],
    'credit_score': [700, 680, 720, 650, 710, 690],
    'loan_amount': [5000, 10000, 7000, 8000, 6000, 9000],
    'loan_intent_Personal': [1,0,1,0,1,0],
    'loan_intent_Education': [0,1,0,1,0,1]
})

X_cols = ['credit_score', 'loan_amount', 'loan_intent_Personal', 'loan_intent_Education']
X = sm.add_constant(sample_data[X_cols].astype(float))
y = sample_data['loan_status']


def test_get_classification_rate():
    model = sm.Logit(y, X).fit(disp=0)
    rate, cutoff = get_classification_rate(model, X_cols, df=sample_data, y='loan_status')
    
    # Check type and reasonable ranges
    assert isinstance(rate, float)
    assert isinstance(cutoff, float)
    assert 0 <= rate <= 100
    assert 0 <= cutoff <= 1


def test_best_alpha():
    alphas = [0.1, 1, 10]
    best_model, best_classification, best_alpha_val = best_alpha(
        y, X, alphas, lp='l1', df=sample_data, y_var='loan_status'
    )
    
    # Check types
    assert best_model is not None
    assert isinstance(best_classification, (tuple, list))
    assert isinstance(best_alpha_val, (float, np.float64))
    
    # Check ranges
    assert 0 <= best_classification[0] <= 100
    assert 0 <= best_classification[1] <= 1
