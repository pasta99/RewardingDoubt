import math

scale = 5.0
max_reward = -0.0010005003335835344
min_reward = -6.907755278982137 / 2
wrong_format_penalty = -scale * 3.0

def reward_function(confidence: int, is_answer_correct: bool) -> float:
    if confidence == None or confidence > 10 or confidence < 0:
        return wrong_format_penalty
    
    normalized_confidence = min(0.999, max(0.001, confidence / 10))

    if is_answer_correct:
        score = math.log(normalized_confidence)
    else: 
        score = math.log(1 - normalized_confidence)

    norm_score = (score - min_reward) / (max_reward - min_reward)
    if is_answer_correct:
        norm_score += 0.25
    return float(scale * norm_score)