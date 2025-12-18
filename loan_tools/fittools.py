import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings

def get_classification_rate(model, X_cols, df, y='loan_status'):
    df = df.copy()
    X_pred = df[X_cols].astype(float)
    df['loan_prob'] = model.predict_proba(X_pred)[:, 1]
    
    cutoffs = np.linspace(0, 1, 101)
    scores = {
        cutoff: (df['loan_prob'] >= cutoff).eq(df[y]).mean()
        for cutoff in cutoffs
    }
    best_cutoff = max(scores, key=scores.get)
    best_classification_rate = scores[best_cutoff]
    
    return best_classification_rate * 100, round(best_cutoff, 2)


def best_alpha(alphas, train, test, lp='l1', y_var='loan_status'):
    y_train = train[y_var]
    X_cols_reg = [col for col in train.columns if col != y_var]
    X_train = train[X_cols_reg].astype(float)
    
    best_model = None
    best_classification = [0, 0]
    best_alpha = 0
    
    for alpha in alphas:
        C = 1 / alpha
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LogisticRegression(
                penalty=lp,
                C=C,
                solver='liblinear' if lp == 'l1' else 'lbfgs',
                max_iter=1000,
                random_state=42
            )
            model.fit(X_train, y_train)
        
        classification_rate = get_classification_rate(model, X_cols_reg, df=test, y=y_var)
        
        if classification_rate[0] > best_classification[0]:
            best_model = model
            best_classification = classification_rate
            best_alpha = alpha
    
    return best_model, best_classification, best_alpha