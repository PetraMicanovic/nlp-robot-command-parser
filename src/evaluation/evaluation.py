"""
Module for model evaluation and error analysis.

This module provides:
    - comparing constrained vs. unconstrained decoding (exact match)
    - analysing accuracy grouped by command length
    - printing formatted evaluation reports
"""

from collections import defaultdict
from src.models.t5_model import predict


def evaluate_model(test_data, model, tokenizer, cfg, device, n=200):
    """
    Evaluates the model on the first n examples of the test set.
    Runs unconstrained decoding for comparison.

    Parameters:
    test_data: list of dict
        Each element has {"commands": str, "actions": str}
    model: T5ForConditionalGeneration
    tokenizer: T5Tokenizer
    cfg: dict
        Full config loaded from config.json
    device: str
    n: int
        Number of examples to evaluate

    Returns:
    results: dict
        Keys: constrained_exact_match, unconstrained_exact_match, n_evaluated
    """
    model_cfg = cfg["model"]
    exact = 0

    for i in range(n):
        ex = test_data[i]
        cmd = ex["commands"]
        gold = ex["actions"].strip()

        pred = predict(
            cmd,
            model,
            tokenizer,
            prefix=model_cfg["prefix"],
            max_input_len=model_cfg["max_input_len"],
            max_target_len=model_cfg["max_target_len"],
            device=device,
            num_beams=model_cfg["num_beams"],
        ).strip()

        if pred == gold:
            exact += 1

    return {
        "exact_match": round(exact / n, 4),
        "n_evaluated": n,
    }


def analyse_by_length(test_data, model, tokenizer, cfg, device, n=500):
    """
    Computes exact match accuracy grouped by command length (buckets of 3 tokens).

    Parameters:
    test_data: list of dict
    model: T5ForConditionalGeneration
    tokenizer: T5Tokenizer
    cfg: dict
    device: str
    n: int
        Number of examples to analyse

    Returns:
    buckets: dict
        Keys are bucket start values (int).
        Values are dicts with "correct" and "total" counts.
    """
    model_cfg = cfg["model"]
    buckets = defaultdict(lambda: {"correct": 0, "total": 0})

    for i in range(n):
        ex = test_data[i]
        cmd = ex["commands"]
        gold = ex["actions"].strip()
        length = len(cmd.split())
        bucket = (length // 3) * 3

        pred = predict(
            cmd,
            model,
            tokenizer,
            prefix=model_cfg["prefix"],
            max_input_len=model_cfg["max_input_len"],
            max_target_len=model_cfg["max_target_len"],
            device=device,
            num_beams=model_cfg["num_beams"],
        ).strip()

        buckets[bucket]["total"] += 1
        if pred == gold:
            buckets[bucket]["correct"] += 1

    return dict(buckets)


def print_exact_match(results):
    """
    Prints exact match.

    Parameters:
    results: dict
        Output of evaluate_model()
    """
    print(f"  Еxact match : {results['exact_match']:.2%}")
    print(f"  (evaluated on {results['n_evaluated']} examples)")


def print_length_analysis(buckets):
    """
    Prints a table of accuracy by command length bucket.

    Parameters:
    buckets: dict
        Output of analyse_by_length()
    """
    print("Accuracy by command length:")
    print(f"  {'Length':>8}  {'Accuracy':>10}  {'Examples':>10}")
    print("  " + "-" * 33)
    for b in sorted(buckets.keys()):
        s = buckets[b]
        if s["total"] > 0:
            acc = s["correct"] / s["total"]
        else:
            acc = 0.0
        print(f"  {str(b)+'-'+str(b+2):>8}  {acc:>10.2%}  {s['total']:>10}")
