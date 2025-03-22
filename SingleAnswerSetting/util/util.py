import torch

def remove_padding(tensor, pad_token):
    start_idx = 0
    # while start_idx < len(tensor) and tensor[start_idx] == pad_token:
    #     start_idx += 1

    # Find the end index where padding starts again
    end_idx = len(tensor) - 1
    while end_idx >= 0 and tensor[end_idx] == pad_token:
        end_idx -= 1

    # Slice the tensor to remove padding, add 1 to end_idx to include the last non-pad token
    trimmed_tensor = tensor[start_idx:end_idx+1]
    return trimmed_tensor

def change_confidence(tokens, confidences_tokens, new_token):
    for t in confidences_tokens:
        tokens = torch.where(tokens == t, new_token, tokens)
    return tokens