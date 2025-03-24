import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns

from util.ResponseHandling import load_MultiFactResults
from util.Metrics import MultiFactResults_to_accuracy, Metric, MultiFactResults_to_calibration_curve, MultiFactResults_to_histogram_confidences, MultiFactResults_to_calibration_curve_first_answer_only, MultiFactResults_to_ECE, MultiFactResults_to_auroc_score

def create_responses_if_necessary(model: str, dataset: str, split: str, use_cached: bool, is_unsloth: bool, additional_info={}) -> str:
    cache_name = f"{dataset}_{split}_results_multianswer.json"
    cache_dir = os.path.join(model, cache_name)
    
    if not os.path.isfile(cache_dir) or not use_cached:
        from InferenceDataset import inference_dataset
        inference_dataset(model, is_unsloth, dataset, split, cache_dir)

    return cache_dir

def get_results(model: str, dataset: str, split: str, use_cached: bool, is_unsloth: bool, additional_info={}):
    results_dir = create_responses_if_necessary(model, dataset, split, use_cached, is_unsloth, additional_info=additional_info)   
    results = load_MultiFactResults(results_dir)

    return results

def get_calibration_curve(model: str, dataset: str, split: str, metric = Metric.F1, threshold: float = 0.5, max_confidence: float = 5, use_cached: bool = True, is_unsloth: bool = False, label_probs: bool = False, additional_info={}, legend=True, fine_tuned=False):

    results = get_results(model, dataset, split, use_cached, is_unsloth, additional_info=additional_info)

    x, y = MultiFactResults_to_calibration_curve(results, metric=metric, threshold=threshold, max_confidence=max_confidence)
    if label_probs:
        for i in range(len(y)):
            plt.text(x[i], y[i], f'{y[i]:.3f}', ha='left', va='bottom')
    
    # Plot model's calibration curve
    plt.plot(x, y, marker = '.', label="Calibration curve")
    if legend:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return plt

def get_ideal_calibration_curve(legend=True):
    plt.plot([0, 1], [0, 1], linestyle = '--', label="Ideal")
    if legend:
        plt.legend()
    return plt

def get_ECE(model: str, dataset: str, split: str, metric = Metric.F1, threshold: float = 0.5, max_confidence: float = 5, n_bins: int = 11, use_cached: bool = True, is_unsloth: bool = False, additional_info={}):
    results = get_results(model, dataset, split, use_cached, is_unsloth, additional_info=additional_info)

    return MultiFactResults_to_ECE(results, metric=metric, max_confidence=max_confidence, n_bins=n_bins, threshold=threshold)

def get_auroc(model: str, dataset: str, split: str, metric = Metric.F1, threshold: float = 0.5, max_confidence: float = 5, use_cached: bool = True, is_unsloth: bool = False, additional_info={}):
    results = get_results(model, dataset, split, use_cached, is_unsloth, additional_info=additional_info)

    return MultiFactResults_to_auroc_score(results, metric=metric, max_confidence=max_confidence, threshold=threshold)

def get_histogram_of_confidences(model: str, dataset: str, split: str, max_confidence: int = 5, use_cached: bool = True, is_unsloth: bool = False, additional_info={}):

    results = get_results(model, dataset, split, use_cached, is_unsloth, additional_info=additional_info)
    counts, bins = MultiFactResults_to_histogram_confidences(results, max_confidence=max_confidence)

    data = pd.DataFrame({
            'x': max_confidence * np.arange(0, 1.1, 0.1),         # Values from 0.0 to 1.0 in 0.1 steps
            'y': counts  # Example y-values
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

def get_accuracy(model: str, dataset: str, split: str, metric = Metric.F1, threshold: float = 0.5, max_confidence: float = 10, use_cached: bool = True, is_unsloth: bool = False, additional_info={}):
    results = get_results(model, dataset, split, use_cached, is_unsloth, additional_info=additional_info)

    accuracy = MultiFactResults_to_accuracy(results, metric, threshold, max_confidence)

    return accuracy.item()


def get_all_metrics(model: str, dataset: str, split: str, metric = Metric.F1, threshold: float = 0.5, max_confidence: float = 5, n_bins: int = 11, use_cached: bool = True, is_unsloth: bool = False, additional_info={}):
    
    ece = get_ECE(model, dataset, split, metric, threshold, max_confidence, n_bins, use_cached, is_unsloth, additional_info=additional_info)
    accuracy = get_accuracy(model, dataset, split, metric, threshold, max_confidence, use_cached, is_unsloth, additional_info=additional_info)
    auroc = get_auroc(model, dataset, split, metric, threshold, max_confidence, use_cached, is_unsloth, additional_info=additional_info)

    metrics = {"Model": model, "Dataset": dataset, "Split": split, "Metric": metric.name, "Threshold": threshold, "ECE": ece, "Accuracy": accuracy, "AUROC": auroc}

    return metrics
