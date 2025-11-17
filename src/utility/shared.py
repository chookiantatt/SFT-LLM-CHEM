from pathlib import Path
import json
import os
from transformers.trainer_utils import get_last_checkpoint

# Inference batch sizes
models_to_batch_size_mapping = {
    "meta-llama/Llama-2-13b-chat-hf": 50,
    "codellama/CodeLlama-34b-hf": 25,
    "mistralai/Mixtral-8x7B-v0.1": 20,
    "tiiuae/falcon-40b": 20,
    "01-ai/Yi-34B-200K": 25,
}


def ensure_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path


def find_best_model(output_dir):
    if os.path.isdir(output_dir):
        model_listdir = os.listdir(output_dir)
        if "full_eval_window.json" in model_listdir:
            with open(f"{output_dir}/full_eval_window.json", "r") as fmod:
                finetune_trainer_logs = json.load(fmod)
            bestmodel_checkpoint = str(finetune_trainer_logs["best_model_checkpoint"])
            return bestmodel_checkpoint
        elif "trainer_state.json" in model_listdir:
            with open(f"{output_dir}/trainer_state.json", "r") as fmod:
                finetune_trainer_logs = json.load(fmod)
            bestmodel_checkpoint = str(finetune_trainer_logs["best_model_checkpoint"])
            return bestmodel_checkpoint
        else:
            last_checkpoint = get_last_checkpoint(output_dir)
            if last_checkpoint is not None:
                with open(f"{last_checkpoint}/trainer_state.json", "r") as fmod:
                    finetune_trainer_logs = json.load(fmod)
                bestmodel_checkpoint = str(
                    finetune_trainer_logs["best_model_checkpoint"]
                )
                return bestmodel_checkpoint
            else:
                raise Exception(
                    "No checkpoints available in output dir, please train model"
                )
    else:
        raise Exception(f"output dir ({output_dir}) not found, please train model")
