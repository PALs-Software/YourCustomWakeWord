from tqdm import tqdm
import numpy as np


def create_roc_curve_fp_rate(predictions, no_of_points = 50):

    prediction_len = len(predictions)
    fp_rate_per_separating_point = []
    for separating_point in tqdm(np.linspace(0.01, 0.99, num=no_of_points)):
        decisions = [1 if p > separating_point else 0 for p in predictions]
        false_positives = sum(decisions)
        fp_rate_per_separating_point.append(false_positives / prediction_len)

    return fp_rate_per_separating_point


def create_roc_curve_fn_rate(predictions, no_of_points = 50):

    prediction_len = len(predictions)
    fn_rate_per_separating_point = []
    for separating_point in tqdm(np.linspace(0.01, 0.99, num=no_of_points)):
        decisions = [1 if p < separating_point else 0 for p in predictions]
        false_negatives = sum(decisions)
        fn_rate_per_separating_point.append(false_negatives / prediction_len)

    return fn_rate_per_separating_point