from .utils import plot_settings


def box_plots(models, dataframe, dependent, independent):
    plot_settings()
    feature_dataframe = dataframe[[independent, dependent]]
    feature_dataframe.boxplot(by=independent)
    return None
