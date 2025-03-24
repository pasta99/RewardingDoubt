prompts = {
        "mc": 
            "You will get test questions with possible options. Answer with the correct option. Additionally provide a confidence between 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, of how sure you are the answer is correct. A value close to 0 means you think there is a high probability that the answer is wrong. The closer the value is to 10, the higher you think is the probability that the answer is correct. The output should have the format 'Answer: <answer_index>, Confidence: <confidence>' and nothing else.",
        "open": 
            "You will get questions. Answer with the correct answer. Additionally provide a confidence between 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, of how sure you are the answer is correct. A value close to 0 means you think there is a high probability that the answer is wrong. The closer the value is to 10, the higher you think is the probability that the answer is correct. The output should have the format 'Answer: <answer>, Confidence: <confidence>' and nothing else.",
        "context_open": 
            "You will get a context with a question. Answer with the correct answer. Additionally provide a confidence between 0, 1, 2, 3, 4, 5. A value close to 0 means you think there is a high probability that you could be wrong. The closer the value is to 5, the lower you think is the chance that you could be wrong. The output should have the format 'Answer: <answer>, Confidence: <confidence>' and nothing else.",
        "multifact":
            """Instructions: 
            1. You will get a question with multiple possible answers.
            2. Enumerate all possible answers you know. After each individual answer state your confidence in this answer. The format should be 'Answer: <answer>, Confidence: <confidence>\n' for each individual answer. 
            3. The confidence should be an integer number between 0 and 10. 0 means you know for certain the answer is wrong. 10 means you know for certain the answer is correct. 
            4. Do not say anything else. Do not write multiple answers in one answer block.
            5. When asked about dates, answer with the specific year.
            """
}

def get_prompt(type: str):
    try:
        prompt = prompts[type]
    except KeyError:
        raise NotImplementedError(f"For this task type no prompt has been specified! : {type}")
    
    return prompt