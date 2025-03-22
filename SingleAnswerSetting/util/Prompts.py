prompts = {
    "mc": 
        "You will get test questions with possible options. Answer with the correct option. Additionally provide a confidence between 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, of how sure you are the answer is correct. A value close to 0 means you think there is a high probability that the answer is wrong. The closer the value is to 10, the higher you think is the probability that the answer is correct. The output should have the format 'Answer: <answer_index>, Confidence: <confidence>' and nothing else.",
    "open": 
        "You will get questions. Answer with the correct answer. Additionally provide a confidence between 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, of how sure you are the answer is correct. A value close to 0 means you think there is a high probability that the answer is wrong. The closer the value is to 10, the higher you think is the probability that the answer is correct. The output should have the format 'Answer: <answer>, Confidence: <confidence>' and nothing else.",
    "context_open": 
        "You will get a context with a question. Answer with the correct answer. Additionally provide a confidence between 0, 1, 2, 3, 4, 5. A value close to 0 means you think there is a high probability that you could be wrong. The closer the value is to 5, the lower you think is the chance that you could be wrong. The output should have the format 'Answer: <answer>, Confidence: <confidence>' and nothing else."
}

def get_prompt(task_type: str):
    try:
        prompt = prompts[task_type]
    except KeyError:
        raise NotImplementedError(f"For this task no prompt has been specified! : {type}")
    
    return prompt