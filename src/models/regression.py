import statsmodels.api as sm
from patsy import dmatrices


def ols_regression(independent, dependent, dataframe):
    formula = ''.join([dependent, '~', independent])
    y, X = dmatrices(formula, data=dataframe, return_type='dataframe')

    # Statistica analysis
    mod = sm.OLS(y, X)  # Describe model
    res = mod.fit()  # Fit model
    return res
