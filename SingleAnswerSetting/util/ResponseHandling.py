from dataclasses import dataclass
import dataclasses
from typing import List, Optional
import re
import json
from dacite import from_dict as dict_to_class
import numpy as np
import math
import torch
import torch.nn.functional as F

from util.util import remove_padding


@dataclass
class QAResult:
    question: str
    prediction: Optional[str]
    confidence: Optional[float]
    gt_candidates: List[str]
    
    is_multiple_choice: bool
    is_wrong_format: bool

def QAResult_to_dict(result: QAResult) -> dict:
    return dataclasses.asdict(result)

def dict_to_QAResult(result: dict) -> QAResult:
    return dict_to_class(QAResult, result)

def parse_answer_confidence(output, is_multiple_choice):
    """
    Looks for answer and confidence values with regex. \n
    pattern = r"Answer:\s*(?P<answer>.*?)\s*Confidence:\s*(?P<confidence>\d+)"

    Returns:
        Tuple[str, int]: answer, confidence
        is None, None if pattern not found
    """
    pattern = r"Answer:\s*(?P<answer>.*?),\s*Confidence:\s*(?P<confidence>\d+)"

    # Search for the pattern in the input string
    match = re.search(pattern, output)

    # Check if a match is found
    if match:
        # Extract the answer and confidence
        answer = match.group("answer")
        confidence = int(match.group("confidence"))
        if is_multiple_choice:
            if ":" in answer:
                splitted_answer = answer.split(":")
                if not (len(splitted_answer[0]) == 1):
                    return None, None
                else:
                    answer = splitted_answer[0]
        
        return answer, confidence
    else:
        return None, None

def response_to_QAResult(question: str, response: str, gt_candidates: List[str], is_multiple_choice: bool = False) -> QAResult:
    prediction, confidence = parse_answer_confidence(response, is_multiple_choice=is_multiple_choice)
    is_wrong_format = prediction == None

    result = QAResult(question, prediction, confidence, gt_candidates, is_multiple_choice, is_wrong_format)
    return result

def save_QAResults(results: List[QAResult], out_dir: str):
    results_dict = [QAResult_to_dict(res) for res in results]

    with open(out_dir, 'w') as fout:
        json.dump(results_dict , fout, indent=4)

def load_QAResults(read_dir: str) -> List[QAResult]:
    with open(read_dir, "r") as f:
        results_dicts = json.load(f) 

    results = [dict_to_QAResult(r) for r in results_dicts]
    return results