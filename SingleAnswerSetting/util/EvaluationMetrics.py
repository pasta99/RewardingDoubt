from collections import Counter
import string
import re
from enum import Enum
from typing import List, Optional, Tuple
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
import numpy as np
from torchmetrics.classification import BinaryCalibrationError
import torch
from scipy import stats

from util.ResponseHandling import QAResult

class Metric(Enum):
    EXACT = 1
    F1 = 2

# Taken from official TriviaQA evaluation code 
# https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

# Taken from official TriviaQA evaluation code 
# https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# Adapted from official TriviaQA evaluation code 
# https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
def exact_match_score(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0

def f1_score_aliases(prediction: Optional[str], aliases: List[str]) -> float:
    """ Compares prediction to all possible aliases and returns maximum F1 score """
    if prediction == None:
        return 0.0
    scores = [f1_score(prediction, a) for a in aliases]
    return max(scores)

def exact_match_aliases(prediction: Optional[str], aliases: List[str]) -> float:
    if prediction == None:
        return 0.0
    scores = [exact_match_score(prediction, a) for a in aliases]
    return max(scores)

def is_answer_correct(prediction: Optional[str], aliases: List[str], metric: Metric, threshold: float, is_mc: bool = False) -> bool:

    if is_mc:
        parts = prediction.split(":")
        if len(parts) > 1:
            prediction = parts[0]

    result = 0
    if metric == Metric.EXACT:
        result = exact_match_aliases(prediction, aliases)
    elif metric == Metric.F1:
        result = f1_score_aliases(prediction, aliases)

    return result > threshold

def QAResults_to_labels_probs(results: List[QAResult], metric: Metric, threshold: float = 0.5, max_confidence: int = 10) -> Tuple[float, float]:
    results_filtered = [res for res in results if not res.is_wrong_format]
    labels = [1 if is_answer_correct(res.prediction, res.gt_candidates, metric, threshold, res.is_multiple_choice) else 0 for res in results_filtered]
    probabilities = [res.confidence / max_confidence for res in results_filtered]

    return labels, probabilities

def QAResults_to_calibration_curve(results: List[QAResult], metric: Metric, threshold: float = 0.5, max_confidence: int = 10):
    labels, probabilities = QAResults_to_labels_probs(results, metric, threshold, max_confidence)

    x, y = calibration_curve(labels, probabilities, n_bins = 11)
    return y, x

def QAResults_to_ECE(results: List[QAResult], metric: Metric, threshold: float = 0.5, max_confidence: int = 10, n_bins: int = 11):
    results_filtered = [res for res in results if not res.is_wrong_format]
    labels = torch.tensor([1 if is_answer_correct(res.prediction, res.gt_candidates, metric, threshold) else 0 for res in results_filtered])
    probabilities = torch.tensor([res.confidence / max_confidence for res in results_filtered])

    metric = BinaryCalibrationError(n_bins=n_bins, norm='l1')
    ece = metric(probabilities, labels)
    return ece.item()

def QAResults_to_brier_score(results: List[QAResult], metric: Metric, threshold: float = 0.5, max_confidence: int = 10) -> float:
    labels, probabilities = QAResults_to_labels_probs(results, metric, threshold, max_confidence)

    brier = brier_score_loss(labels, probabilities)
    return brier

def QAResults_to_auroc_score(results: List[QAResult], metric: Metric, threshold: float = 0.5, max_confidence: int = 10) -> float:
    labels, probabilities = QAResults_to_labels_probs(results, metric, threshold, max_confidence)

    auroc = roc_auc_score(labels, probabilities)
    return auroc

def QAResults_to_roc_curve(results: List[QAResult], metric: Metric, threshold: float = 0.5, max_confidence: int = 10) -> float:
    labels, probabilities = QAResults_to_labels_probs(results, metric, threshold, max_confidence)

    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    return fpr, tpr, thresholds

def QAResults_to_histogram_confidences(results: List[QAResult], max_confidence: int = 10):
    confidences = [res.confidence for res in results if not (res.confidence == None)]

    counts, bins = np.histogram(confidences, bins=11, range=[-0.1 * max_confidence, 1.1 * max_confidence])

    return counts, bins

def QAResults_to_pearson_correlation(results: List[QAResult], metric: Metric, threshold: float = 0.5, max_confidence: int = 10):

    x, y = QAResults_to_calibration_curve(results, metric, threshold, max_confidence)

    res = stats.pearsonr(x, y)

    coeff = res.statistic
    pvalue = res.pvalue

    return coeff, pvalue

def QAResults_to_accuracy(results: List[QAResult], metric: Metric, threshold: float = 0.5, max_confidence: int = 10):
    labels, _ = QAResults_to_labels_probs(results, metric, threshold, max_confidence)

    return np.mean(labels)