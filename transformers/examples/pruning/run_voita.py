# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""


import dataclasses
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch

from transformers import AutoConfig, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
    GatedBertForSequenceClassification,
    AdamW,
)
from torch.utils.data import DataLoader, SequentialSampler, Subset
from tqdm import tqdm
from pruning_utils import print_2d_tensor, convert_gate_to_mask


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class PruningArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    l0_penalty: float = field(
        default=0.5, metadata={"help": "coefficient for l0 regularization."}
    )
    num_of_heads: Optional[int] = field(
        default=None, metadata={"help": "number of heads to be kept."}
    )
    joint_pruning: Optional[bool] = field(
        default=False, metadata={"help": "if set, train gate variables and other parameters in the original model together."}
    )
    pruning_lr: Optional[float] = field(
        default=0.1, metadata={"help": "learning rate for gate variables."}
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, PruningArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, pruning_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, pruning_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    if data_args.task_name == "mnli":
        data_args.task_name="mnli-mm"
        eval_dataset = (
            GlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
            if training_args.do_eval
            else None
        )
        data_args.task_name = "mnli"
    else:
        eval_dataset = (
            GlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
            if training_args.do_eval
            else None
        )

    if data_args.task_name == "mnli":
        metric = "eval_mnli/acc"
    else:
        metric = "eval_acc"

    torch.manual_seed(42)
    model = GatedBertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    model.apply_gates(pruning_args.l0_penalty)

    if pruning_args.joint_pruning:
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not "log_a" in n],
                "lr": training_args.learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters() if "log_a" in n],
                "lr": pruning_args.pruning_lr,
            },
        ]
    else:
        for n, p in model.named_parameters():
            if not "log_a" in n:
                p.requires_grad = False
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if "log_a" in n],
                "lr": pruning_args.pruning_lr,
            },
        ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Initialize our Trainer
    training_args.max_steps = -1
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
        optimizers=(optimizer, None)
    )

    # Training
    trainer.train()
    trainer.save_model()

    gates = model.get_gate_values()
    if pruning_args.num_of_heads is None:
        head_mask = convert_gate_to_mask(gates)
    else:
        head_mask = convert_gate_to_mask(gates, pruning_args.num_of_heads)
    model.remove_gates()
    model.apply_masks(head_mask)
    score = trainer.evaluate(eval_dataset=eval_dataset)[metric]
    print_2d_tensor(head_mask)
    logger.info("l0 penalty: {}, remaining heads: {}, accuracy: {}".format(pruning_args.l0_penalty, head_mask.sum(), score * 100))


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
