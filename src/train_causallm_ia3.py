import os
import argparse
import math
import torch
import numpy as np
from itertools import chain
from dotenv import load_dotenv

from huggingface_hub import login
from datasets import load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed,
    TrainerCallback,
)
from peft import (
    PeftModel,
    get_peft_model,
    IA3Config,
    TaskType,
)

from transformers.trainer_utils import get_last_checkpoint
import evaluate


# Argument Parser
def get_args():
    parser = argparse.ArgumentParser(description="IA3 Training Script")

    parser.add_argument(
        "--pretrained_model", required=True, default="meta-llama/Llama-2-13b-chat-hf"
    )
    parser.add_argument("--gpu", default="0,1,2,3")
    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--trainval_dataset_dir", default="")
    parser.add_argument("--test_dataset_dir", default="")
    parser.add_argument("--ft_task_type", default="gen")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_steps", type=int, default=16000)
    parser.add_argument("--save_steps", type=int, default=1600)
    parser.add_argument("--block_size", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--cache_dir", default="/app/llm_misc/cache_dir")

    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--overwrite_output_dir", action="store_true")

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_seval", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")

    parser.add_argument("--disable_bfloat16", action="store_true")
    parser.add_argument("--disable_bestmodel", action="store_true")
    parser.add_argument("--checkpoint_num", default="")

    parser.add_argument(
        "--do_rope_scaling", default=None, choices=[None, "linear", "dynamic"]
    )

    return parser.parse_args()


# Utility
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# Early Stopping Callback
class EarlyStopOnLossIncrease(TrainerCallback):
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.prev_loss = np.inf

    def on_evaluate(self, args, state, control, **kwargs):
        logs = state.log_history
        if not logs:
            return

        current_loss = logs[-1].get("eval_loss")
        if current_loss is None:
            return

        if current_loss > self.prev_loss:
            self.counter += 1
        else:
            self.counter = 0

        self.prev_loss = current_loss
        print(f"[EarlyStopping] Loss increase streak: {self.counter}")

        if self.counter >= self.patience:
            control.should_training_stop = True


# Dataset Helper
def group_texts(examples, block_size):
    concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = (len(concatenated["input_ids"]) // block_size) * block_size

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# Main
def main():
    args = get_args()
    print(f"ARGS: {args}")

    # Environment setup
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Login to HF
    load_dotenv()
    hf_auth = os.getenv("hf_auth")
    login(token=hf_auth)

    # Reproducibility
    SEED = 1
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    set_seed(SEED)

    # Output directory
    output_dir = f"/app/llm_misc/train_scripts/saved_models/cpt-ia3-casuallm/{args.pretrained_model}/{args.dataset_name}/"
    ensure_dir(output_dir)

    # Load dataset
    if args.do_train or args.do_eval or args.do_seval:
        trainval_path = (
            args.trainval_dataset_dir
            or f"../datasets/{args.ft_task_type}/{args.dataset_name}/clm_tokenized/{args.pretrained_model}"
        )
        encoded_trainval = load_from_disk(trainval_path)
        print(f"Loaded TrainVal from {trainval_path}")

    if args.do_test:
        test_path = (
            args.test_dataset_dir
            or f"../datasets/{args.ft_task_type}/patent_level_abstract_test/clm_tokenized/{args.pretrained_model}"
        )
        encoded_test = load_from_disk(test_path)
        print(f"Loaded Test from {test_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        padding_side="left",
        cache_dir=args.cache_dir,
        use_auth_token=hf_auth,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model config
    config = AutoConfig.from_pretrained(
        args.pretrained_model, cache_dir=args.cache_dir, use_auth_token=hf_auth
    )

    # → Rope Scaling
    if args.do_rope_scaling:
        config.rope_scaling = {"type": args.do_rope_scaling, "factor": 4}
        print(f"Using ROPE scaling: {config.rope_scaling}")

    # Identify IA3 target modules
    if "llama" in args.pretrained_model.lower():
        target_modules = ["k_proj", "v_proj", "down_proj"]
        feedforward = ["down_proj"]
    elif "falcon" in args.pretrained_model.lower():
        target_modules = ["self_attention"]
        feedforward = ["mlp"]
    else:
        target_modules = ["query", "value", "key"]
        feedforward = ["key"]

    # Load Model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model,
        cache_dir=args.cache_dir,
        use_auth_token=hf_auth,
        device_map="auto",
        trust_remote_code=True,
        config=config,
    )

    # PEFT / IA3 config
    if args.do_train:
        print("Applying IA3...")
        peft_config = IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            feedforward_modules=feedforward,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Load best checkpoint for eval/test
    elif not args.disable_bestmodel:
        from utility.shared import find_best_model

        ckpt = (
            output_dir + f"checkpoint-{args.checkpoint_num}"
            if args.checkpoint_num
            else find_best_model(output_dir)
        )
        print(f"Loading best PEFT checkpoint: {ckpt}")
        model = PeftModel.from_pretrained(
            model, ckpt, device_map="auto", use_auth_token=hf_auth
        )

    # Mixed precision
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        evaluation_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.save_steps,
        optim="adamw_torch",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=not args.disable_bfloat16,
    )

    # Group datasets
    if args.do_train:
        train_dataset = encoded_trainval["train"].map(
            lambda x: group_texts(x, args.block_size),
            batched=True,
            batch_size=10000,
        )

    if args.do_seval:
        seval_dataset = encoded_trainval["seval"].map(
            lambda x: group_texts(x, args.block_size),
            batched=True,
            batch_size=10000,
        )

    if args.do_eval:
        eval_dataset = encoded_trainval["eval"].map(
            lambda x: group_texts(x, args.block_size),
            batched=True,
        )

    if args.do_test:
        test_dataset = encoded_test["test"].map(
            lambda x: group_texts(x, args.block_size),
            batched=True,
        )

    # Metrics
    metric = evaluate.load("accuracy", cache_dir=args.cache_dir)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=(
            seval_dataset if args.do_seval else (eval_dataset if args.do_eval else None)
        ),
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[EarlyStopOnLossIncrease(patience=10)] if args.do_train else None,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # TRAIN
    if args.do_train:
        last_ckpt = None
        if os.path.isdir(output_dir) and not args.overwrite_output_dir:
            last_ckpt = get_last_checkpoint(output_dir)
            if last_ckpt and not args.resume_from_checkpoint:
                raise ValueError(
                    f"Found checkpoint {last_ckpt} but resume_from_checkpoint=False.\n"
                    "Use --resume_from_checkpoint or --overwrite_output_dir."
                )

        trainer.train(resume_from_checkpoint=last_ckpt)
        trainer.save_model()
        trainer.save_state()
        print("Training done.")

    # EVAL
    if args.do_eval:
        metrics = trainer.evaluate()
        metrics["perplexity"] = math.exp(metrics["eval_loss"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        print("Eval complete.")

    # TEST
    if args.do_test:
        metrics = trainer.evaluate(test_dataset)
        metrics["perplexity"] = math.exp(metrics["eval_loss"])
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)
        print("Test complete.")

    print("DONE.")


if __name__ == "__main__":
    main()
