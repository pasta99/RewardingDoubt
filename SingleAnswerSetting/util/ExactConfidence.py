"""Exact-gradient alternative to the PPO step: the confidence is one of 11 strings ": k<eot>", so
its expected reward can be computed exactly instead of estimated from one sampled response.

For each answer that ends in " Confidence", score the 11 completions with the policy and with the
reference (the LoRA adapter disabled, as TRL's reference for PEFT models):
    J  = sum_k p(k) R(k) + (1 - sum_k p(k)) * wrong_format_penalty
    KL = KL(p || p_ref) over the 12 outcomes (the 11 levels and "anything else")
    loss = -(J - beta * KL),   beta = the PPO trainer's adaptive KL coefficient
R(k) is QAResult_to_reward for confidence k, exactly as Train.py rewards a sampled response, so the
reward, format penalty, KL controller, optimizer, LoRA and data are all unchanged; only the gradient
estimator differs. No value head, advantage estimate or sampling of the confidence is needed.
The update loop mirrors PPO's: ppo_epochs passes over the batch in minibatches of mini_batch_size.

Two ways to compute it. --objective exact (fast=True) shares the work: the 11 strings ": k" differ
only in the number, so one forward pass over prefix + ": " gives every level's probability from one
softmax (when "10" is two tokens, as in Qwen, the sequence also gets "1" appended, so the next
position gives P("0" | "1") and splits level 1 from level 10). One sequence per answer instead of
eleven, and the whole minibatch in one call. The approximation: p(k) = P(": k") leaves out the end-
of-turn factor, so "other" means malformed before the number, and stopping after the number is not
trained (it is near-certain for this model and format; check the evaluation's wrong_format_rate).

--objective exact_reference (fast=False) scores the 11 complete strings ": k<eot>" as separate
sequences: the same objective without the approximation, about five times slower; useful for
checking the fast path.
"""
import math
import random

import torch

from util.ResponseHandling import QAResult, response_to_QAResult
from util.RLHelper import QAResult_to_reward, wrong_format_penalty


def end_of_turn(tokenizer):
    """Llama-3's <|eot_id|>, or the tokenizer's EOS where that token does not exist (Qwen: <|im_end|>)."""
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    return eot if eot is not None and eot != tokenizer.unk_token_id else tokenizer.eos_token_id


def _level_logps(lm, prefix, candidates, pad):
    """log P(candidate | prefix) for each candidate token list, one batched forward."""
    seqs = [prefix + c for c in candidates]
    width = max(map(len, seqs))
    ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs], device=next(lm.parameters()).device)
    logits = lm(input_ids=ids).logits
    out = []
    for b, c in enumerate(candidates):
        span = logits[b, len(prefix) - 1:len(prefix) - 1 + len(c)].float().log_softmax(-1)
        out.append(span.gather(-1, torch.tensor(c, device=next(lm.parameters()).device)[:, None]).sum())
    return torch.stack(out)


def _shared_level_logps(lm, prefixes, levels, pad):
    """log P(": k" | prefix) for all 11 levels and every prefix, one forward pass (fast=True)."""
    n_common = 0
    while all(len(lv) > n_common + 1 for lv in levels) and len({lv[n_common] for lv in levels}) == 1:
        n_common += 1
    common, ends = levels[0][:n_common], [lv[n_common:] for lv in levels]
    two_token_ten = len(ends[10]) == 2  # Qwen: "10" is "1" "0"
    first = [e[0] for e in ends]
    seqs = [p + common + ([ends[10][0]] if two_token_ten else []) for p in prefixes]
    width = max(map(len, seqs))
    device = next(lm.parameters()).device
    logits = lm(input_ids=torch.tensor([s + [pad] * (width - len(s)) for s in seqs], device=device)).logits
    rows = []
    for b, p in enumerate(prefixes):
        n = len(p) + n_common
        logp = logits[b, len(p) - 1:n + 1].float().log_softmax(-1)
        common_logp = sum(logp[i, t] for i, t in enumerate(common))
        level = common_logp + logp[n_common, torch.tensor(first, device=device)]
        if two_token_ten:
            p_zero = logp[n_common + 1, ends[10][1]]
            level = torch.cat([level[:1], level[1:2] + torch.log1p(-p_zero.exp().clamp(max=1 - 1e-7)), level[2:10],
                               level[1:2] + p_zero])
        rows.append(level)
    return torch.stack(rows)


def exact_confidence_step(ppo_trainer, tokenizer, predictions, prompt_lengths, questions, gt_candidates,
                          is_multiple_choice, rng=random.Random(0), fast=False):
    """One exact update on a batch of generated answers. Returns, per scored answer, its expected reward,
    KL to the reference and mean stated confidence (sum_k k p(k) / sum_k p(k)), before the update."""
    model, config = ppo_trainer.model, ppo_trainer.config
    lm = model.pretrained_model
    eot = end_of_turn(tokenizer)
    confidence = tokenizer.convert_tokens_to_ids("ĠConfidence")
    candidates = [tokenizer.encode(f": {k}", add_special_tokens=False) + [eot] for k in range(11)]
    levels = [c[:-1] for c in candidates]
    rows = []
    for prediction, n, question, gt, mc in zip(predictions, prompt_lengths, questions, gt_candidates, is_multiple_choice):
        prefix = prediction.tolist()
        if prefix[-1] != confidence:
            continue  # the answer never reached " Confidence": every completion is malformed, no gradient
        answer = response_to_QAResult(question, tokenizer.decode(prefix[n:], skip_special_tokens=True) + ": 0", gt, mc)
        if answer.prediction is None:
            continue
        rewards = torch.tensor([QAResult_to_reward(QAResult(question, answer.prediction, k, gt, mc, False))
                                for k in range(11)], device=next(lm.parameters()).device)
        if fast:
            rows.append([prefix, rewards, None])
        else:
            with torch.no_grad(), lm.disable_adapter():
                reference = _level_logps(lm, prefix, candidates, tokenizer.pad_token_id)
            rows.append((prefix, rewards, reference))
    if not rows:
        return []
    if fast:  # the reference for the whole batch in one call
        with torch.no_grad(), lm.disable_adapter():
            for row, reference in zip(rows, _shared_level_logps(lm, [r[0] for r in rows], levels, tokenizer.pad_token_id)):
                row[2] = reference
    beta = ppo_trainer.kl_ctl.value

    def objective(prefix, rewards, reference, logp=None):
        if logp is None:
            logp = _level_logps(lm, prefix, candidates, tokenizer.pad_token_id)
        p, p_ref = logp.exp(), reference.exp()
        p_other, p_other_ref = (1 - p.sum()).clamp(min=1e-8), (1 - p_ref.sum()).clamp(min=1e-8)
        expected = (p * rewards).sum() + p_other * wrong_format_penalty
        kl = (p * (logp - reference)).sum() + p_other * (p_other.log() - p_other_ref.log())
        return expected, kl, (p * torch.arange(11, device=p.device)).sum() / p.sum()

    model.train()
    stats = []
    for epoch in range(config.ppo_epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for start in range(0, len(order), config.mini_batch_size):
            ppo_trainer.optimizer.zero_grad()
            chunk = order[start:start + config.mini_batch_size]
            shared = _shared_level_logps(lm, [rows[i][0] for i in chunk], levels, tokenizer.pad_token_id) if fast else [None] * len(chunk)
            losses = []
            for i, logp in zip(chunk, shared):
                expected, kl, confidence = objective(*rows[i], logp=logp)
                losses.append(-(expected - beta * kl) / len(chunk))
                if epoch == 0:
                    stats.append(dict(expected=expected.item(), kl=kl.item(), confidence=confidence.item()))
                if not fast:  # one backward per answer keeps the 11-sequence graphs small
                    losses[-1].backward()
            if fast:
                torch.stack(losses).sum().backward()
            ppo_trainer.optimizer.step()
    ppo_trainer.kl_ctl.update(sum(s["kl"] for s in stats) / len(stats), config.batch_size)
    return stats
