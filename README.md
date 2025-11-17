# SFT-LLM-CHEM
## Overview

This repository contains scripts for preparing datasets, tokenizing text, and training Large Language Models (LLMs) using Continued Pretraining (CPT) and IA3 parameter-efficient fine-tuning. It also includes utilities for evaluation and inference.

## Main Components
1. Dataset Preparation

Clean text and prepare training data.

Tokenize datasets into HF DatasetDict format for Causal LM training.

Supports large-scale chunking and EOS padding.

Scripts:

tokenize_clm_dataset.py

chem_utils.py

2. Training
Continued Pretraining (CPT)

Improves domain understanding before fine-tuning.

Works with LLaMA, Mistral, Falcon, and other CausalLM models.

IA3 Fine-Tuning

Lightweight PEFT method for domain-specific generation tasks.

Supports checkpoint resuming, best-checkpoint selection, and optional ROPE scaling.

Scripts:

train_causallm_cpt.py

train_causallm_ia3.py

3. Evaluation & Inference

Evaluate checkpoints using perplexity and validation loss.

Generate text using CPT/IA3-tuned models.

Basic Usage
Tokenization
python tokenize_clm_dataset.py --dataset_name <name> --pretrained_model <model>

CPT Training
python train_causallm_cpt.py --do_train --dataset_name <name> --pretrained_model <model>

IA3 Fine-Tuning
python train_causallm_ia3.py --do_train --dataset_name <name> --pretrained_model <model>
