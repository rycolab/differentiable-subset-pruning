import logging

import numpy as np
import torch
from tqdm import tqdm

from transformers import glue_compute_metrics, EvalPrediction
from typing import Dict, List, Tuple
from seqeval.metrics import f1_score, precision_score, recall_score

from typing import Callable, Dict, Optional

from transformers import (
    glue_compute_metrics,
)

logger = logging.getLogger(__name__)

def align_predictions(predictions: np.ndarray, label_ids: np.ndarray, label_map) -> Tuple[List[int], List[int]]:
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != torch.nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list, out_label_list

def compute_metrics(p: EvalPrediction, label_map) -> Dict:
    preds_list, out_label_list = align_predictions(p.predictions, p.label_ids, label_map)
    return {
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }

def evaluate(args, model, eval_dataloader, head_mask=None):
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads

    if head_mask is None:
        head_mask = torch.ones(n_layers, n_heads).to(args.device)

    model.apply_masks(head_mask)
    # Evaluate
    preds = None
    labels = None
    for step, inputs in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
        for k, v in inputs.items():
            inputs[k] = v.to(args.device)

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(**inputs)
        loss, logits, all_attentions = (
            outputs[0],
            outputs[1],
            outputs[-1],
        )  # Loss and logits are the first, attention the last

        # Also store our logits/labels if we want to compute metrics afterwards
        if preds is None:
            preds = logits.detach().detach().cpu().numpy()
            labels = inputs["labels"].detach().detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().detach().cpu().numpy(), axis=0)
            labels = np.append(labels, inputs["labels"].detach().detach().cpu().numpy(), axis=0)
    
    if args.task_name in ['ner', 'pos']:
        score = compute_metrics(EvalPrediction(predictions=preds, label_ids=labels), model.config.id2label)['f1']
    else:
        preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
        score = glue_compute_metrics(args.task_name, preds, labels)[args.metric_name]
        
    return score

def compute_heads_importance(
    args, model, train_dataloader, head_mask=None
):
    """ This method shows how to compute:
        - head importance scores according to http://arxiv.org/abs/1905.10650
    """
    # Prepare our tensors
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(args.device)

    if head_mask is None:
        head_mask = torch.ones(n_layers, n_heads).to(args.device)
    head_mask.requires_grad_(requires_grad=True)
    model.apply_masks(head_mask)
    
    tot_tokens = 0.0
    for step, inputs in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
        for k, v in inputs.items():
            inputs[k] = v.to(args.device)

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(**inputs)
        loss, logits, all_attentions = (
            outputs[0],
            outputs[1],
            outputs[-1],
        )  # Loss and logits are the first, attention the last
        
        first_derivative = torch.autograd.grad(loss, head_mask)[0]
        head_importance += first_derivative.abs().detach()

        tot_tokens += inputs["attention_mask"].float().detach().sum().data

    # Normalize
    head_importance /= tot_tokens
    # Layerwise importance normalization
    if not args.dont_normalize_importance_by_layer:
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

    if not args.dont_normalize_global_importance:
        head_importance = (head_importance - head_importance.min()) / (head_importance.max() - head_importance.min())

    return head_importance

def mask_heads(
    args, model, train_dataloader, eval_dataloader
):
    """ This method shows how to mask head (set some heads to zero), to test the effect on the network,
        based on the head importance scores, as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    head_importance = compute_heads_importance(args, model, train_dataloader)
    original_score = evaluate(args, model, eval_dataloader)
    logger.info("Pruning: original score: %f", original_score)

    new_head_mask = torch.ones_like(head_importance)
    args.num_of_heads

    scores = []
    sparsities = []
    all_head_masks = []

    current_score = original_score
    sparsities.append(0.0)
    scores.append(current_score * 100)
    all_head_masks.append(new_head_mask.clone())

    # while current_score >= original_score * args.masking_threshold:
    while new_head_mask.sum() != 0:

        head_mask = new_head_mask.clone()  # save current head mask
        # heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 0.0] = float("Inf")
        current_heads_to_mask = head_importance.view(-1).sort()[1]

        if len(current_heads_to_mask) <= args.num_of_heads:
            break

        # mask heads
        current_heads_to_mask = current_heads_to_mask[:args.num_of_heads]
        logger.info("Heads to mask: %s", str(current_heads_to_mask.tolist()))
        new_head_mask = new_head_mask.view(-1)
        new_head_mask[current_heads_to_mask] = 0.0
        new_head_mask = new_head_mask.view_as(head_mask)
        print_2d_tensor(new_head_mask)

        # Compute metric and head importance again
        if args.exact_pruning and new_head_mask.sum() != 0:
            head_importance = compute_heads_importance(
                args, model, train_dataloader, head_mask=new_head_mask.clone()
            )
        current_score = evaluate(args, model, eval_dataloader, head_mask=new_head_mask.clone())

        sparsity = 100 - new_head_mask.sum() / new_head_mask.numel() * 100
        scores.append(current_score * 100)
        sparsities.append(sparsity)
        all_head_masks.append(new_head_mask.clone())

        logger.info(
            "Masking: current score: %f, remaning heads %d (%.1f percents)",
            current_score,
            new_head_mask.sum(),
            100 - sparsity,
        )


    logger.info("Final head mask")
    print_2d_tensor(new_head_mask)

    return scores, sparsities, all_head_masks

def print_2d_tensor(tensor):
    """ Print a 2D tensor """
    logger.info("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
        else:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))

def convert_gate_to_mask(gates, num_of_heads=None):
    if num_of_heads is not None:
        head_mask = torch.zeros_like(gates)
        current_heads_to_keep = gates.view(-1).sort(descending = True)[1]
        current_heads_to_keep = current_heads_to_keep[:num_of_heads]
        head_mask = head_mask.view(-1)
        head_mask[current_heads_to_keep] = 1.0
        head_mask = head_mask.view_as(gates)
    else:
        head_mask = (gates > 0.5).float()
    return head_mask