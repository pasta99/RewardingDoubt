from collections import Counter
import string
import re
import math
from enum import Enum
from typing import List, Optional, Tuple
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
import numpy as np
from torchmetrics.classification import BinaryCalibrationError
import torch
from scipy import stats
import itertools

from util.MultiFactResult import MultiFactResult

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

def are_answers_correct(answers, gt_candidates, metric: Metric, threshold: float):
    correct = [is_answer_correct(a, gt_candidates, metric, threshold, False) for a in answers]

    return correct

def MultiFactResults_to_labels_probs(results: List[MultiFactResult], metric: Metric, threshold: float = 0.5, max_confidence: int = 5) -> Tuple[float, float]:
    corrects_answers = [are_answers_correct(result.predictions, result.gt_candidates, metric, threshold) for result in results]
    all_answers = list(itertools.chain.from_iterable(corrects_answers))
    labels = [1 if a else 0 for a in all_answers]

    confidences = [result.confidences for result in results]
    all_confidences = list(itertools.chain.from_iterable(confidences))

    labels, all_confidences = zip(*[(l, c) for l, c in zip(labels, all_confidences) if not math.isnan(c)])

    probabilities = [confidence / max_confidence for confidence in all_confidences]

    return labels, probabilities

def MultiFactResults_to_labels_probs_first_answer_only(results: List[MultiFactResult], metric: Metric, threshold: float = 0.5, max_confidence: int = 5) -> Tuple[float, float]:
    corrects_answers = [are_answers_correct(result.predictions, result.gt_candidates, metric, threshold) for result in results]
    corrects_answers = [a for a in corrects_answers if len(a) > 0]
    all_answers = [a[0] for a in corrects_answers]
    labels = [1 if a else 0 for a in all_answers]

    confidences = [result.confidences for result in results]
    confidences = [c for c in confidences if len(c) > 0]
    all_confidences = [c[0] for c in confidences]
    probabilities = [confidence / max_confidence for confidence in all_confidences]

    return labels, probabilities

def MultiFactResults_to_calibration_curve(results: List[MultiFactResult], metric: Metric, threshold: float = 0.5, max_confidence: int = 5):
    labels, probabilities = MultiFactResults_to_labels_probs(results, metric, threshold, max_confidence)

    x, y = calibration_curve(labels, probabilities, n_bins = 11)
    return y, x
def MultiFactResults_to_calibration_curve_first_answer_only(results: List[MultiFactResult], metric: Metric, threshold: float = 0.5, max_confidence: int = 5):
    labels, probabilities = MultiFactResults_to_labels_probs_first_answer_only(results, metric, threshold, max_confidence)

    x, y = calibration_curve(labels, probabilities, n_bins = 11)
    return y, x

def MultiFactResults_to_ECE(results: List[MultiFactResult], metric: Metric, threshold: float = 0.5, max_confidence: int = 5, n_bins: int = 11):
    labels, probabilities = MultiFactResults_to_labels_probs(results, metric, threshold, max_confidence)

    labels = torch.tensor(labels)
    probabilities = torch.tensor(probabilities)

    metric = BinaryCalibrationError(n_bins=n_bins, norm='l1')
    ece = metric(probabilities, labels)
    return ece.item()

# def QAResults_to_brier_score(results: List[QAResult], metric: Metric, threshold: float = 0.5, max_confidence: int = 5) -> float:
#     labels, probabilities = QAResults_to_labels_probs(results, metric, threshold, max_confidence)

#     brier = brier_score_loss(labels, probabilities)
#     return brier

def MultiFactResults_to_auroc_score(results: List[MultiFactResult], metric: Metric, threshold: float = 0.5, max_confidence: int = 5) -> float:
    labels, probabilities = MultiFactResults_to_labels_probs(results, metric, threshold, max_confidence)

    auroc = roc_auc_score(labels, probabilities)
    return auroc.item()

# def QAResults_to_roc_curve(results: List[QAResult], metric: Metric, threshold: float = 0.5, max_confidence: int = 5) -> float:
#     labels, probabilities = QAResults_to_labels_probs(results, metric, threshold, max_confidence)

#     fpr, tpr, thresholds = roc_curve(labels, probabilities)
#     return fpr, tpr, thresholds

def MultiFactResults_to_histogram_confidences(results: List[MultiFactResult], max_confidence: int = 5):
    confidences = [result.confidences for result in results]
    all_confidences = list(itertools.chain.from_iterable(confidences))

    counts, bins = np.histogram(all_confidences, bins=11, range=[-0.1 * max_confidence, 1.1 * max_confidence])

    return counts, bins

# def QAResults_to_pearson_correlation(results: List[QAResult], metric: Metric, threshold: float = 0.5, max_confidence: int = 5):

#     x, y = QAResults_to_calibration_curve(results, metric, threshold, max_confidence)

#     res = stats.pearsonr(x, y)

#     coeff = res.statistic
#     pvalue = res.pvalue

#     return coeff, pvalue

def MultiFactResults_to_accuracy(results: List[MultiFactResult], metric: Metric, threshold: float = 0.5, max_confidence: int = 5):
    labels, _ = MultiFactResults_to_labels_probs(results, metric, threshold, max_confidence)

    return np.mean(labels)