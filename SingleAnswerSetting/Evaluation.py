from util.ResponseHandling import load_QAResults
from util.EvaluationMetrics import QAResults_to_calibration_curve, Metric, QAResults_to_histogram_confidences, QAResults_to_ECE, QAResults_to_pearson_correlation, QAResults_to_accuracy, QAResults_to_auroc_score

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns

def create_responses_if_necessary(model: str, dataset: str, split: str, use_cached: bool, is_unsloth: bool) -> str:
    cache_name = f"{dataset}_{split}_results_singlefact.json"
    cache_dir = os.path.join(model, cache_name)
    
    if not os.path.isfile(cache_dir) or not use_cached:
        from InferenceDatasetSplit import inference_dataset

        inference_dataset(model, is_unsloth, dataset, split, cache_dir)

    return cache_dir

def get_results(model: str, dataset: str, split: str, use_cached: bool, is_unsloth: bool):
    results_dir = create_responses_if_necessary(model, dataset, split, use_cached, is_unsloth)   
    results = load_QAResults(results_dir)

    return results

def get_calibration_curve(model: str, dataset: str, split: str, metric = Metric.F1, threshold: float = 0.5, max_confidence: float = 5, use_cached: bool = True, is_unsloth: bool = False, label_probs: bool = False, legend=True, fontsize=12):

    results = get_results(model, dataset, split, use_cached, is_unsloth)

    x, y = QAResults_to_calibration_curve(results, metric=metric, threshold=threshold, max_confidence=max_confidence)

    if label_probs:
        for i in range(len(y)):
            plt.text(x[i], y[i], f'{y[i]:.3f}', ha='left', va='bottom')
    
    # Plot model's calibration curve
    plt.plot(x, y, marker = '.', label="Calibration Curve")
    if legend:
        plt.legend(fontsize=fontsize, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel("Average predicted probability per bin", fontsize=fontsize)
    plt.ylabel("Ratio of positives",fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    return plt

def get_ideal_calibration_curve(legend=True, fontsize=12):
    plt.plot([0, 1], [0, 1], linestyle = '--', label="Ideal")
    if legend:
        plt.legend(fontsize=fontsize)
    return plt

def get_ECE(model: str, dataset: str, split: str, metric = Metric.F1, threshold: float = 0.5, max_confidence: float = 5, n_bins: int = 11, use_cached: bool = True, is_unsloth: bool = False):
    results = get_results(model, dataset, split, use_cached, is_unsloth)

    return QAResults_to_ECE(results, metric=metric, max_confidence=max_confidence, n_bins=n_bins, threshold=threshold)

def get_auroc(model: str, dataset: str, split: str, metric = Metric.F1, threshold: float = 0.5, max_confidence: float = 5, use_cached: bool = True, is_unsloth: bool = False):
    results = get_results(model, dataset, split, use_cached, is_unsloth)

    return QAResults_to_auroc_score(results, metric=metric, max_confidence=max_confidence, threshold=threshold)

def get_histogram_of_confidences(model: str, dataset: str, split: str, max_confidence: int = 5, use_cached: bool = True, is_unsloth: bool = False):

    results = get_results(model, dataset, split, use_cached, is_unsloth)
    counts, bins = QAResults_to_histogram_confidences(results, max_confidence=max_confidence)

    data = pd.DataFrame({
            'x': max_confidence * np.arange(0, 1.1, 0.1),
            'y': counts
        })

    fig, ax1 = plt.subplots()
    sns.barplot(x='x', y='y', data=data, ax=ax1, color='lightblue', edgecolor='black')

    plt.xlabel("Bins of predicted confidences")
    plt.ylabel("Bin size")
    plt.title("")

    plt.tick_params(axis='both', labelsize=14)
    xticks = max_confidence * np.arange(0, 1.1, 0.1)
    ax1.set_xticklabels([f"{tick:.1f}" if i % 2 == 0 else "" for i, tick in enumerate(xticks)])
            
    return plt

def get_accuracy(model: str, dataset: str, split: str, metric = Metric.F1, threshold: float = 0.5, max_confidence: float = 5, use_cached: bool = True, is_unsloth: bool = False):
    results = get_results(model, dataset, split, use_cached, is_unsloth)

    accuracy = QAResults_to_accuracy(results, metric, threshold, max_confidence)

    return accuracy


def get_all_metrics(model: str, dataset: str, split: str, metric = Metric.F1, threshold: float = 0.5, max_confidence: float = 5, n_bins: int = 11, use_cached: bool = True, is_unsloth: bool = False):
    
    ece = get_ECE(model, dataset, split, metric, threshold, max_confidence, n_bins, use_cached, is_unsloth)
    accuracy = get_accuracy(model, dataset, split, metric, threshold, max_confidence, use_cached, is_unsloth)
    auroc = get_auroc(model, dataset, split, metric, threshold, max_confidence, use_cached, is_unsloth)

    metrics = {"Model": model, "Dataset": dataset, "Split": split, "Metric": metric.name, "Threshold": threshold, "ECE": ece, "Accuracy": accuracy, "AUROC": auroc}

    return metrics
