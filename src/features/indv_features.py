from data.extract_data import read_individual_diff


def extract_individual_features(config, subject):
    indv_data = read_individual_diff(config, subject)
    return indv_data
