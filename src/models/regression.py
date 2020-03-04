import statsmodels.formula.api as smf


def ols_regression(independent, dependent, dataframe):
    formula = ''.join([dependent, '~', independent])

    # Statistica analysis
    mod = smf.ols(formula, data=dataframe)  # Describe model
    res = mod.fit()  # Fit model
    print(res.summary())
    return res
