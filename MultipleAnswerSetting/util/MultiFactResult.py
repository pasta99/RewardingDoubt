from typing import List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class MultiFactResult:
    question: str
    response: str

    predictions: Optional[List[str]]
    confidences: Optional[List[float]]
    gt_candidates: List[str]