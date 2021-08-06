import argparse
import shutil
import time
from json import JSONDecodeError
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


logger = getLogger(__name__)

try:
    from .utils import (
        Seq2SeqDataset,
        calculate_bleu,
        calculate_rouge,
        lmap,
        load_json,
        parse_numeric_cl_kwargs,
        save_json,
        use_task_specific_params,
        write_txt_file,
    )
except ImportError:
    from utils import (
        Seq2SeqDataset,
        calculate_bleu,
        calculate_rouge,
        lmap,
        load_json,
        parse_numeric_cl_kwargs,
        save_json,
        use_task_specific_params,
        write_txt_file,
    )


def eval_data_dir(
    data_dir,
    save_dir: str,
    model_name: str,
    bs: int = 8,
    max_source_length: int = 1024,
    type_path="val",
    n_obs=None,
    fp16=False,
    num_beams: int = 4,
    task="summarization",
    local_rank=None,
    **generate_kwargs,
) -> Dict:
    """Run evaluation on part of the data for one gpu and save to {save_dir}/rank_{rank}_output.json"""
    model_name = str(model_name)
    assert local_rank is not None
    torch.distributed.init_process_group(backend="nccl", rank=local_rank)

    save_dir = Path(save_dir)
    save_path = save_dir.joinpath(f"rank_{local_rank}_output.json")
    torch.cuda.set_device(local_rank)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).cuda()
    if fp16:
        model = model.half()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Inferred tokenizer type: {tokenizer.__class__}")  # if this is wrong, check config.model_type.
    use_task_specific_params(model, task)  # update config with task specific params
    if max_source_length is None:
        max_source_length = tokenizer.model_max_length
    ds = Seq2SeqDataset(
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length=1024,
        type_path=type_path,
        n_obs=n_obs,
        prefix=model.config.prefix,
    )
    sampler = ds.make_sortish_sampler(bs, distributed=True, add_extra_examples=False)
    data_loader = DataLoader(ds, sampler=sampler, batch_size=bs, collate_fn=ds.collate_fn)
    dec_kwargs = dict(skip_special_tokens=True, clean_up_tokenization_spaces=False)  # tokenizer.decode
    results = []
    for batch in tqdm(data_loader):
        summaries = model.generate(
            input_ids=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
            num_beams=num_beams,
            **generate_kwargs,
        )
        preds = tokenizer.batch_decode(summaries, **dec_kwargs)
        labels = tokenizer.batch_decode(batch["labels"], **dec_kwargs)
        ids = batch["ids"]
        for i in range(len(labels)):
            label, pred = labels[i], preds[i]
            results.append(dict(pred=pred, label=label, id=ids[i].item()))
    save_json(results, save_path)
    return results, sampler.num_replicas


def run_generate():
    parser = argparse.ArgumentParser(
        epilog="Unspecified args like --num_beams=2 --decoder_start_token_id=4 are passed to model.generate"
    )
    parser.add_argument("--data_dir", type=str, help="like cnn_dm/test.source")
    parser.add_argument(
        "--model_name",
        type=str,
        help="like facebook/bart-large-cnn,t5-base, etc.",
        default="sshleifer/distilbart-xsum-12-3",
    )
    parser.add_argument("--save_dir", type=str, help="where to save", default="tmp_gen")
    parser.add_argument("--max_source_length", type=int, default=None)
    parser.add_argument(
        "--type_path", type=str, default="test", help="which subset to evaluate typically train/val/test"
    )
    parser.add_argument("--reference_path", type=str, required=False, help="like cnn_dm/test.target")
    parser.add_argument("--task", type=str, default="summarization", help="used for task_specific_params + metrics")
    parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
    parser.add_argument(
        "--local_rank", type=int, default=-1, required=False, help="should be passed by distributed.launch"
    )

    parser.add_argument(
        "--n_obs", type=int, default=None, required=False, help="How many observations. Defaults to all."
    )
    parser.add_argument(
        "--sync_timeout",
        type=int,
        default=600,
        required=False,
        help="How long should master process wait for other processes to finish.",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--debug", action="store_true")
    start_time = time.time()
    args, rest = parser.parse_known_args()
    generate_kwargs = parse_numeric_cl_kwargs(rest)
    if generate_kwargs:
        print(f"parsed the following generate kwargs: {generate_kwargs}")
    json_save_dir = Path(args.save_dir + "_tmp")
    Path(json_save_dir).mkdir(exist_ok=True)  # this handles locking.
    intermediate_files = list(json_save_dir.glob("rank_*.json"))
    if intermediate_files:
        raise ValueError(f"Found files at {json_save_dir} please move or remove them.")
        # In theory, a node could finish and save before another node hits this. If this happens, we can address later.

    Path(args.save_dir).mkdir(exist_ok=True)
    results, num_replicas = eval_data_dir(
        args.data_dir,
        json_save_dir,
        args.model_name,
        type_path=args.type_path,
        batch_size=args.bs,
        fp16=args.fp16,
        task=args.task,
        local_rank=args.local_rank,
        n_obs=args.n_obs,
        max_source_length=args.max_source_length,
        **generate_kwargs,
    )

    if args.local_rank <= 0:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(exist_ok=True)
        partial_results = gather_results_from_each_node(num_replicas, json_save_dir, args.sync_timeout)
        preds, labels = combine_partial_results(partial_results)
        # Calculate metrics, save metrics,  and save _generations.txt
        calc_bleu = "translation" in args.task
        score_fn = calculate_bleu if calc_bleu else calculate_rouge
        metric_name = "bleu" if calc_bleu else "rouge"
        metrics: Dict = score_fn(preds, labels)
        metrics["n_obs"] = len(preds)
        runtime = time.time() - start_time
        metrics["seconds_per_sample"] = round(runtime / metrics["n_obs"], 2)
        # TODO(@stas00): add whatever metadata to metrics
        metrics_save_path = save_dir.joinpath(f"{args.type_path}_{metric_name}.json")
        save_json(metrics, metrics_save_path)
        print(metrics)
        write_txt_file(preds, save_dir.joinpath(f"{args.type_path}_generations.txt"))
        if args.debug:
            write_txt_file(labels, save_dir.joinpath(f"{args.type_path}.target"))
        else:
            shutil.rmtree(json_save_dir)


def combine_partial_results(partial_results) -> Tuple[List, List]:
    """Concatenate partial results into one file, then sort it by id."""
    records = []
    for partial_result in partial_results:
        records.extend(partial_result)
    records = list(sorted(records, key=lambda x: x["id"]))
    preds = [x["pred"] for x in records]
    labels = [x["label"] for x in records]
    return preds, labels


def gather_results_from_each_node(num_replicas, save_dir, timeout) -> List[Dict[str, List]]:
    # WAIT FOR lots of .json files
    start_wait = time.time()
    logger.info("waiting for all nodes to finish")
    json_data = None
    while (time.time() - start_wait) < timeout:
        json_files = list(save_dir.glob("rank_*.json"))
        if len(json_files) < num_replicas:
            continue
        try:
            # make sure all json files are fully saved
            json_data = lmap(load_json, json_files)
            return json_data
        except JSONDecodeError:
            continue
    else:
        raise TimeoutError("Rank 0 gave up on waiting for other processes")
    # Unreachable


if __name__ == "__main__":
    # Usage for MT:
    run_generate()
