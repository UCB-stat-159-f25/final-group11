import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings

def get_classification_rate(md, X, df = test, y = 'loan_status'):
    df = df.copy()

    df[X] = df[X].astype(int)
    X_pred = sm.add_constant(df[X])

    df['loan_prob'] = md.predict(X_pred)
    
    cutoffs = np.linspace(0, 1, 101)
    scores = {
        cutoff: (df['loan_prob'] >= cutoff).eq(df['loan_status']).mean()
        for cutoff in cutoffs
        }

    best_cutoff = max(scores, key=scores.get)
    best_classification_rate = scores[best_cutoff]
    return best_classification_rate * 100, round(best_cutoff, 2)

def best_alpha(y, X_reg, alphas, lp = 'l1', df = train, y_var = 'loan_status'):
    y = df[y_var]
    best_model = None
    best_classification = [0,0]
    best_alpha = 0
    for alpha in alphas:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.Logit(y, X_reg).fit_regularized(method=lp, alpha=alpha)
        classification_rate = get_classification_rate(model, X_cols)
        if classification_rate[0] > best_classification[0]:
            best_model = model
            best_classification = classification_rate
            best_alpha = alpha
    return best_model, best_classification, best_alpha