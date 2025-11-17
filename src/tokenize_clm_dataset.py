import os
import argparse
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from huggingface_hub import login

# Argument Parser
parser = argparse.ArgumentParser(
    description="Tokenize dataset for CLM Fine-Tuning",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--dataset_name", required=True)
parser.add_argument("--ft_task_type", default="gen")
parser.add_argument("--output_dir")
parser.add_argument("--pretrained_model", required=True)
parser.add_argument("--do_train", action="store_true")
parser.add_argument("--do_seval", action="store_true")
parser.add_argument("--do_eval", action="store_true")
parser.add_argument("--do_test", action="store_true")
parser.add_argument(
    "--hf_token", help="HuggingFace token (recommended to pass via CLI)"
)
args = parser.parse_args()

# HF Login
if args.hf_token:
    login(token=args.hf_token)
    print("HF login successful")
else:
    print("No HF token provided; skipping login.")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    args.pretrained_model, cache_dir="./cache_dir"
)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

TRUNCATE_LEN = 10000
print("Tokenizer max length:", TRUNCATE_LEN)

# Files & Output Paths
base_path = f"../datasets/{args.ft_task_type}/{args.dataset_name}"
output_dir = args.output_dir or f"{base_path}/clm_tokenized/{args.pretrained_model}"

dataset_dict = DatasetDict()


# Tokenization Function
def tokenize_function(row):
    result = tokenizer(
        row["cleaned_abstract"],
        truncation=True,
        max_length=TRUNCATE_LEN,
        padding=True,
    )

    # force EOS at end
    if result["input_ids"][-1] != tokenizer.eos_token_id:
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    return result


# Utility Loader
def load_and_tokenize(split: str):
    print(f"Tokenizing {split}...")
    parquet_path = os.path.join(base_path, "raw", split)

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Missing Parquet file for split: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    dataset = Dataset.from_pandas(df)

    # Remove all columns except "patent_id"
    keep_cols = ["patent_id"]
    remove_cols = [c for c in dataset.features.keys() if c not in keep_cols]

    return dataset.map(
        tokenize_function,
        remove_columns=remove_cols,
    )


# Processing Splits
if args.do_train:
    dataset_dict["train"] = load_and_tokenize("train")

if args.do_seval:
    dataset_dict["seval"] = load_and_tokenize("seval")

if args.do_eval:
    dataset_dict["eval"] = load_and_tokenize("eval")

if args.do_test:
    dataset_dict["test"] = load_and_tokenize("test")

# Save
dataset_dict.save_to_disk(output_dir)
print(f"Saved tokenized datasets to: {output_dir}")
