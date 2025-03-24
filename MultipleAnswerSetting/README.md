# RewardingDoubt

## Training
Run the Train.py file. See the parser at the bottom of the file for the arguments. 

## Adding your own dataset 
If you want to train with your own dataset, simply add a DatasetDescriptor to the list in util/DataHelper.py: 

~~~ 
class DatasetDescriptor:
    huggingface_config: str # The arguments to load the dataset in huggingface 
    normalize_function: Callable # Your own custom function to normalize the dataset to a common format
    type: str # The task type ("open", "mc" or "context_open")
    columns_to_remove: List[str] # Which columns to remove from the original dataset
~~~ 

To normalize a dataset each sample has to have the following format:

{"question": question, "answer": correct_answer, "gt_candidates": candidates, "is_multiple_choice": True}

question: The question given to the model without the system prompt. In case it is a multiple choice question the options should be included too. 

answer: The answer for debugging and logging

gt_candidates: A list of all the answers that should be considered correct 

is_multiple_choice: Wether the question is a multiple choice question or not.

This way training code does not have to be adapted. You can check  util/DataHelper.py for examples. 

## Evaluation 
To evaluate a model you can use the code in Evaluation.py or simply use the EvaluateModel notebook. 

Note that the results will be cached in a file stored at the models location. You can also call the evaluation code with the flag to not use the cached version. 

