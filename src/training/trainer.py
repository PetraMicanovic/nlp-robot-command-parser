"""
Module for training the T5 seq2seq model.

This module provides:
    - building training arguments from config
    - defining evaluation metrics (exact match, token accuracy)
    - constructing and returning a Seq2SeqTrainer
"""

import numpy as np
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)


def build_compute_metrics(tokenizer):
    """
    Returns a compute_metrics function for Seq2SeqTrainer.

    Computes:
        - exact_match    : fraction of sequences predicted exactly correct
        - token_accuracy : fraction of individual tokens predicted correctly

    Parameters:
    tokenizer: T5Tokenizer

    Returns:
    compute_metrics: callable
    """
    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        cleaned_preds = []
        for p in decoded_preds:
            cleaned_preds.append(p.strip())

        cleaned_labels = []
        for l in decoded_labels:
            cleaned_labels.append(l.strip())

        # Exact match
        exact = 0
        for i in range(len(cleaned_labels)):
            if cleaned_preds[i] == cleaned_labels[i]:
                exact += 1

        exact_match = exact / len(cleaned_labels)

        # Token accuracy
        token_correct = 0
        token_total = 0

        for i in range(len(cleaned_labels)):
            p_toks = cleaned_preds[i].split()
            l_toks = cleaned_labels[i].split()

            n = len(p_toks)
            if len(l_toks) > n:
                n = len(l_toks)

            j = 0
            while j < len(p_toks) and j < len(l_toks):
                if p_toks[j] == l_toks[j]:
                    token_correct += 1
                j += 1

            token_total += n

        if token_total > 0:
            token_acc = token_correct / token_total
        else:
            token_acc = 0.0

        return {
            "exact_match": round(exact_match, 4),
            "token_accuracy": round(token_acc, 4),
        }

    return compute_metrics


def build_trainer(model, tokenizer, tokenized_dataset, training_cfg, device_fp16):
    """
    Builds and returns a configured Seq2SeqTrainer.

    Parameters:
    model: T5ForConditionalGeneration
    tokenizer: T5Tokenizer
    tokenized_dataset: DatasetDict
        Must have "train" and "test" splits
    training_cfg: dict
        Training hyperparameters from config.json["training"]
    device_fp16: bool
        Whether to use mixed precision (True on GPU)

    Returns:
    trainer: Seq2SeqTrainer
    """
    total_steps   = (len(tokenized_dataset["train"]) // training_cfg["train_batch_size"]) * training_cfg["num_epochs"]
    warmup_steps  = int(total_steps * training_cfg["warmup_ratio"])

    args = Seq2SeqTrainingArguments(
        output_dir=training_cfg["output_dir"],
        num_train_epochs=training_cfg["num_epochs"],
        per_device_train_batch_size=training_cfg["train_batch_size"],
        per_device_eval_batch_size=training_cfg["eval_batch_size"],
        learning_rate=training_cfg["learning_rate"],
        warmup_steps= warmup_steps,
        weight_decay=training_cfg["weight_decay"],
        predict_with_generate=True,
        generation_max_length=128,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=200,
        fp16=device_fp16,
        seed=training_cfg["seed"],
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=8
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    return trainer