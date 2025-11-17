import os
import argparse
import sys
import math

import torch
import numpy as np
from itertools import chain
from dotenv import load_dotenv
from datasets import load_from_disk
import evaluate

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from utility.shared import ensure_dir, find_best_model
from huggingface_hub import login

# Arguments
parser = argparse.ArgumentParser(
    description="Causal LM Full FT",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--pretrained_model", type=str, required=True)
parser.add_argument("--gpu", type=str, default="0,1,2,3")
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--trainval_dataset_dir", type=str, default="")
parser.add_argument("--test_dataset_dir", type=str, default="")
parser.add_argument("--ft_task_type", type=str, default="gen")
parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--eval_steps", type=int, default=1000)
parser.add_argument("--save_steps", type=int, default=1000)
parser.add_argument("--block_size", type=int, default=500)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--cache_dir", type=str, default="/app/llm_misc/cache_dir")
parser.add_argument("--resume_from_checkpoint", action="store_true")
parser.add_argument("--overwrite_output_dir", action="store_true")
parser.add_argument("--do_train", action="store_true")
parser.add_argument("--do_seval", action="store_true")
parser.add_argument("--do_eval", action="store_true")
parser.add_argument("--do_test", action="store_true")
parser.add_argument("--disable_bestmodel", action="store_true")
parser.add_argument("--checkpoint_num", default="")

args = parser.parse_args()
print(f"ARGS: {args}")

# Environment Setup
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

load_dotenv()
hf_auth = os.getenv("hf_auth")

login(token=hf_auth)
print("Logged in to HuggingFace.")

sys.path.append("/app/llm_misc")

SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

# Paths
model_name = args.pretrained_model
block_size = args.block_size
cache_dir = args.cache_dir

output_dir = f"/app/llm_misc/train_scripts/saved_models/cpt-casuallm-full/{model_name}/{args.dataset_name}/"
ensure_dir(output_dir)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side="left",
    cache_dir=cache_dir,
    use_auth_token=hf_auth,
)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Device Map (manual)
device_map_dict = {
    **{f"model.layers.{i}": i // 10 for i in range(40)},
    "model.embed_tokens": 0,
    "model.norm": 3,
    "lm_head": 3,
}


# Load Model
def load_model():
    if args.do_train:
        print("Loading model for TRAINING...")
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            return_dict=True,
            cache_dir=cache_dir,
            use_auth_token=hf_auth,
            device_map=device_map_dict,
            trust_remote_code=True,
        )
    else:
        print("Loading model for EVAL/TEST...")
        ckpt = (
            f"{output_dir}checkpoint-{args.checkpoint_num}"
            if args.checkpoint_num
            else find_best_model(output_dir)
        )
        if not ckpt.startswith("/app/llm_misc"):
            ckpt = "/app/llm_misc/train_scripts/" + ckpt

        print("Best checkpoint:", ckpt)
        return AutoModelForCausalLM.from_pretrained(
            ckpt,
            return_dict=True,
            cache_dir=cache_dir,
            use_auth_token=hf_auth,
            device_map=device_map_dict,
            trust_remote_code=True,
        )


model = load_model()
model.torch_dtype = torch.bfloat16

# HF Trainer Settings
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.num_epochs,
    weight_decay=0.01,
    bf16=True,
    optim="adamw_torch",
    evaluation_strategy="steps",
    logging_strategy="steps",
    save_strategy="steps",
    eval_steps=args.eval_steps,
    save_steps=args.save_steps,
    logging_steps=args.save_steps,
    greater_is_better=False,
    metric_for_best_model="eval_loss",
    save_total_limit=3,
)

# Metrics
metric = evaluate.load("accuracy", cache_dir=cache_dir)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)


def preprocess_logits_for_metrics(logits):
    logits = logits[0] if isinstance(logits, tuple) else logits
    return logits.argmax(dim=-1)


# Grouping Function
def group_texts(examples):
    concatenated = {k: list(chain(*examples[k])) for k in examples}
    total = len(concatenated["input_ids"])

    total = (total // block_size) * block_size
    if total == 0:
        return {}

    result = {
        k: [t[i : i + block_size] for i in range(0, total, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# Load Tokenized Dataset(s)
if args.do_train or args.do_seval or args.do_eval:
    trainval_path = (
        args.trainval_dataset_dir
        or f"../datasets/{args.ft_task_type}/{args.dataset_name}/clm_tokenized/{model_name}"
    )
    encoded_trainval = load_from_disk(trainval_path)
    print("Loaded Train/Val from:", trainval_path)

if args.do_test:
    test_path = (
        args.test_dataset_dir
        or f"../datasets/{args.ft_task_type}/patent_level_abstract_test/clm_tokenized/{model_name}"
    )
    encoded_test = load_from_disk(test_path)
    print("Loaded Test from:", test_path)

# Grouping each split
if args.do_train:
    print("Grouping train dataset...")
    train_dataset = encoded_trainval["train"].map(
        group_texts, batched=True, batch_size=10000
    )

if args.do_seval:
    print("Grouping SEVAL dataset...")
    seval_dataset = encoded_trainval["seval"].map(
        group_texts, batched=True, batch_size=10000
    )

if args.do_eval:
    print("Grouping eval dataset...")
    eval_dataset = encoded_trainval["eval"].map(
        group_texts, batched=True, batch_size=10000
    )

if args.do_test:
    print("Grouping test dataset...")
    test_dataset = encoded_test["test"].map(group_texts, batched=True, batch_size=10000)

# Build Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if args.do_train else None,
    eval_dataset=seval_dataset if args.do_seval else None,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

# TRAIN
if args.do_train:
    print("*** TRAINING ***")

    last_checkpoint = None
    if os.path.isdir(output_dir) and not args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint:
            if args.resume_from_checkpoint:
                print(f"Resuming training at {last_checkpoint}")
            else:
                raise ValueError(
                    f"Found checkpoint at {last_checkpoint}. Use --resume_from_checkpoint or --overwrite_output_dir."
                )

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()
    trainer.save_state()
else:
    print("Skipping Training.")

# EVAL
if args.do_eval:
    print("*** EVALUATION ***")
    metrics = trainer.evaluate()

    metrics["perplexity"] = (
        math.exp(metrics["eval_loss"]) if metrics["eval_loss"] < 50 else float("inf")
    )
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

# TEST
if args.do_test:
    print("*** TEST ***")
    metrics = trainer.evaluate(test_dataset)

    metrics["perplexity"] = (
        math.exp(metrics["eval_loss"]) if metrics["eval_loss"] < 50 else float("inf")
    )
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

print("DONE.")
