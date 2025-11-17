"""
Utility functions for supervised fine-tuning and continual pre-training work
on chemical substance datasets: NER, relation extraction, normalization,
long-tail analysis, LLM prompting, BIO conversions, and evaluation helpers.
"""

import json
import re
import string
import time
from collections import Counter

import cconlleval
import matplotlib.pyplot as plt
import nnereval
import requests
import scipy.stats as stats

# Global config
AUTH_KEYS = ["YOUR", "API", "KEY"]
CURR_AUTH_KEY_IDX = 1

ENGINE_MAPPING = {
    "claude-3-opus": "anthropic",
    "claude-3-haiku": "anthropic",
    "claude-3-sonnet": "anthropic",
    "patent-0.3": "patsnap",
    "patent-copilot-0.35": "patsnap",
    "gpt-3.5-turbo": "openai",
    "gpt-4-turbo": "openai",
}

# Basic helpers


def remove_bos_eos_tokens(text):
    """Strip <s> and </s> from model output."""
    return text.lstrip("<s>").rstrip("</s>")


def calc_len(lst):
    """Return len(list) or None if list is None."""
    return None if lst is None else len(lst)


# ---------------------------------------------------------------------------
# LLM PROMPTING
# ---------------------------------------------------------------------------


def prompt_llm(
    prompt_batch,
    model,
    tokenizer,
    to_tokenize=True,
    max_new_tokens=300,
    model_returns_prompt=True,
):
    """
    Send prompts to a HuggingFace LLM model.generate().
    """
    inputs = (
        tokenizer(prompt_batch, return_tensors="pt", padding=True).to(model.device)
        if to_tokenize
        else {k: v.to(model.device) for k, v in prompt_batch.items()}
    )

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trunc_outputs = []

    if not model_returns_prompt:
        for i in range(len(outputs)):
            # Cut out the prompt portion
            prompt_len = inputs["input_ids"][i].size(0 if to_tokenize else 1)
            trunc_outputs.append(outputs[i][prompt_len:])
    else:
        trunc_outputs = outputs

    decoded = tokenizer.batch_decode(trunc_outputs, skip_special_tokens=True)
    return decoded


def prompt_chatgpt(
    messages,
    model,
    temperature=None,
    gateway_api="http://rd-gateway.patsnap.info/compute/openai_chatgpt_turbo",
):
    """
    Send OpenAI-style chat messages to internal gateway API with retry.
    """
    global CURR_AUTH_KEY_IDX

    payload = {"model": model, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature

    headers = {
        "Content-Type": "application/json",
        "Authorization": AUTH_KEYS[CURR_AUTH_KEY_IDX],
        "X-Ai-Engine": ENGINE_MAPPING[model],
    }

    retry = 10
    ret = None

    while retry > 0:
        response = requests.post(gateway_api, headers=headers, json=payload, timeout=60)

        if response.status_code != 200:
            retry -= 1
            if response.status_code == 429:
                time.sleep(60)  # rate limit
            continue

        parsed = json.loads(response.content)
        if parsed["error_code"] in {400, 429}:
            raise Exception(f"Daily query limit exceeded: {parsed}")

        ret = parsed["data"]["message"]
        break

    return ret


# Extraction helpers (CEM/PRO list extraction)


def extract_pro_cem_list(text):
    """
    Extract pro_list and cem_list from model output containing:
       cem_list = ["...", "..."]
       pro_list = ["...", "..."]
    """
    cem_regex = r"(?:cem_list|cemList)\s*[:=]\s*\[([^\]]+)\]"
    pro_regex = r"(?:pro_list|proList)\s*[:=]\s*\[([^\]]+)\]"

    try:
        cem_match = re.search(cem_regex, text, re.IGNORECASE)
        pro_match = re.search(pro_regex, text, re.IGNORECASE)

        cem_list = None
        pro_list = None

        if cem_match:
            cem_contents = cem_match.group(1)
            cem_list = [x.strip("\"' ") for x in cem_contents.split(",")]

        if pro_match:
            pro_contents = pro_match.group(1)
            pro_list = [x.strip("\"' ") for x in pro_contents.split(",")]

        return pro_list, cem_list

    except Exception as e:
        print("extract_pro_cem_list exception:", e)
        return None, None


# File parsing (for .conll BIO format)


def parse_file(path):
    """
    Parse a CONLL file into tokens, labels, and BIO-formatted lists.
    Sorted by sentence length.
    """
    with open(path, "r") as f:
        data = re.split(r"\n\s*\n", f.read().strip())

    tokens, labels = [], []

    for sent in data:
        sent_tokens, sent_labels = [], []
        for line in sent.split("\n"):
            parts = re.split(r" +", line)
            if len(parts) != 2:
                sent_tokens = []
                break
            tok, lab = parts
            tok = tok if tok else " "
            lab = lab if lab else "O"
            sent_tokens.append(tok)
            sent_labels.append(lab)

        if sent_tokens:
            tokens.append(sent_tokens)
            labels.append(sent_labels)

    bio_lists = []
    for x_words, y_labels in zip(tokens, labels):
        bio_lists.append([f"{x} {y}" for x, y in zip(x_words, y_labels)])

    combined = list(zip(tokens, labels, bio_lists))
    combined_sorted = sorted(combined, key=lambda x: len(x[0]))

    return zip(*combined_sorted)


# Document classification / RE extraction helpers


def extract_doc_class_result(text):
    """
    Map 'entailment' → 0, 'contradiction' → 1.
    """
    match = re.search(r"(entailment|contradiction)", text, re.IGNORECASE)
    if not match:
        return -1
    return 0 if match.group(0).lower() == "entailment" else 1


def extract_re_result(text):
    """Return 1 if '1' appears, 0 if '0' appears, else -1."""
    if "1" in text:
        return 1
    if "0" in text:
        return 0
    return -1


# NER helpers


def convert_bio_to_labels(bio_list):
    """Convert ['token LABEL'] → ['LABEL']."""
    return [x.split(" ")[-1] for x in bio_list]


def replace_BIO(bio_format_list):
    """
    Remove non-CEM/PRO labels from BIO list.
    Others replaced with 'O'.
    """
    new_list = []
    for s in bio_format_list:
        token, label = s.split(" ")
        if label == "O" or label[2:] in {"CEM", "PRO"}:
            new_list.append(s)
        else:
            new_list.append(f"{token} O")
    return new_list


def get_sentence_from_bio(bio_list):
    """Reconstruct sentence from BIO list."""
    sent = ""
    for bio in bio_list:
        tok = bio.split(" ")[0]
        sent += tok if tok in string.punctuation else " " + tok
    return sent


def get_sentence_from_word_tokens(tokens):
    """Reconstruct sentence from raw tokens."""
    sent = ""
    for tok in tokens:
        sent += tok if tok in string.punctuation else " " + tok
    return sent


def get_bio_format(pred_sentence, true_bio):
    """
    Convert LLM-tagged string with <sub>, <pro> tags → BIO list aligned to true tokens.
    """
    pred_bio = []
    remain = pred_sentence.strip()
    ongoing = "O"

    for i, bio in enumerate(true_bio):
        word, true_label = bio.split(" ")

        # Handle tag boundaries
        for tag, lbl in [
            ("<sub>", "B-CEM"),
            ("</sub>", "O"),
            ("<pro>", "B-PRO"),
            ("</pro>", "O"),
        ]:
            if remain.startswith(tag):
                ongoing = lbl
                remain = remain[len(tag) :].strip()

        if remain.startswith(word):
            pred_bio.append(f"{word} {ongoing}")
            if ongoing.endswith("CEM"):
                ongoing = "I-CEM"
            elif ongoing.endswith("PRO"):
                ongoing = "I-PRO"
            remain = remain[len(word) :].strip()
        else:
            # mismatch → stop early
            if i == len(true_bio) - 1 and word == ".":
                pred_bio.append(f"{word} O")
                return pred_bio

            print("\nIncorrect sentence tag.")
            print("Expected token:", word)
            print("Remaining output:", remain)
            print("Expected sentence:", get_sentence_from_bio(true_bio))
            print("Predicted:", pred_sentence)
            return pred_bio

    return pred_bio


def extract_ner_result(text, true_bio):
    """Extract BIO tagging from LLM output."""
    return get_bio_format(text, true_bio)


def compute_ner_metrics(preds_list, true_list):
    """
    Flatten lists, compute NER scores using nnereval + cconlleval.
    """
    instruct_errors = 0
    preds, labs = [], []

    for pred, lab in zip(preds_list, true_list):
        if len(pred) != len(lab):
            instruct_errors += 1
            continue

        preds.extend(
            [p if p in ["B-CEM", "I-CEM", "B-PRO", "I-PRO", "O"] else "O" for p in pred]
        )
        labs.extend(
            [
                label if label in ["B-CEM", "I-CEM", "B-PRO", "I-PRO", "O"] else "O"
                for label in lab
            ]
        )

    assert len(preds) == len(labs)

    eval_labels = ["CEM", "PRO", "O"]
    eval_res = nnereval.score(nnereval.evaluate(eval_labels, preds, labs))

    # conlleval
    rows = [f"{i} {labs[i]} {preds[i]}" for i in range(len(labs))]
    counts = cconlleval.evaluate(rows)
    scores = cconlleval.get_scores(counts)

    macro_f1 = 0
    for k in eval_labels:
        if k in scores:
            eval_res[k]["conlleval_precision"] = scores[k][0] / 100
            eval_res[k]["conlleval_recall"] = scores[k][1] / 100
            eval_res[k]["conlleval_f1"] = scores[k][2] / 100
            macro_f1 += eval_res[k]["conlleval_f1"]

    eval_res["conlleval_macro_f1"] = macro_f1 / len(eval_labels)
    eval_res["conlleval_micro_f1"] = cconlleval.metrics(counts)[0].fscore

    return eval_res, instruct_errors


# Long-tail distribution analysis


def plot_longtail(strings, out_fp):
    """Plot long-tail distribution and save."""
    counts = Counter(strings)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    labels = [x for x, _ in sorted_counts]
    freqs = [c for _, c in sorted_counts]

    plt.figure(figsize=(20, 6))
    plt.bar(labels, freqs)
    plt.xlabel("Entity")
    plt.ylabel("Frequency")
    plt.title("Long-Tail Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.xlim(0, 50)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.5)
    plt.savefig(out_fp, format="jpg")

    print(f"Saved: {out_fp}")
    return sorted_counts


def get_longtail_stats(counts):
    """Compute skewness and kurtosis."""
    if not counts:
        return 0, 0
    return stats.skew(counts), stats.kurtosis(counts)


def print_and_plot_stats(pdf, output_dir, pysoda_prop_norm, subs_dict, ppi_dict):
    """
    Print statistics & long-tail plots for PRO/CEM entities.
    """
    pro_list_col = pdf["pro_list"].tolist()
    cem_list_col = pdf["cem_list"].tolist()

    # Flatten raw lists
    flat_pro = [x for lst in pro_list_col if lst for x in lst]
    flat_cem = [x for lst in cem_list_col if lst for x in lst]

    # Normalization pipeline
    flat_pro_w_offsets = [(0, len(p), p) for p in flat_pro]
    prop_out = pysoda_prop_norm.parse_prediction_list(flat_pro_w_offsets)
    flat_norm_pro = [_d["id"][9:] for _d in prop_out]

    flat_norm_cem_sub = [x for x in flat_cem if x in subs_dict]
    flat_norm_cem_ppi = [x for x in flat_cem if x in ppi_dict]
    flat_norm_cem_total = flat_norm_cem_sub + flat_norm_cem_ppi

    # Plots
    sorted_pro = plot_longtail(flat_pro, f"{output_dir}/entity_dist_pro.jpg")
    sorted_norm_pro = plot_longtail(
        flat_norm_pro, f"{output_dir}/entity_dist_norm_pro.jpg"
    )
    sorted_cem = plot_longtail(flat_cem, f"{output_dir}/entity_dist_cem.jpg")
    sorted_norm_cem_sub = plot_longtail(
        flat_norm_cem_sub, f"{output_dir}/entity_dist_norm_cem_sub.jpg"
    )
    sorted_norm_cem_ppi = plot_longtail(
        flat_norm_cem_ppi, f"{output_dir}/entity_dist_norm_cem_ppi.jpg"
    )
    sorted_norm_cem_total = plot_longtail(
        flat_norm_cem_total, f"{output_dir}/entity_dist_norm_cem_total.jpg"
    )

    # Statistics
    for key in ["PRO", "CEM"]:
        col = key.lower() + "_list"
        error = pdf[col].isnull().sum()
        error_rate = error / len(pdf) * 100
        print(f"{key} None rate: {error}/{len(pdf)} ({error_rate:.4g}%)")

        avg_entities = pdf.dropna(subset=[col])[col].apply(calc_len).mean()
        print(f"{key} avg entities (non-null rows): {avg_entities}")

    print(f"Non-unique norm PRO: {len(flat_norm_pro)}/{len(flat_pro)}")
    print(f"Unique norm PRO: {len(sorted_norm_pro)}/{len(sorted_pro)}")

    print(f"Non-unique norm CEM SUBS_ID: {len(flat_norm_cem_sub)}/{len(flat_cem)}")
    print(f"Non-unique norm CEM PPI_ID: {len(flat_norm_cem_ppi)}/{len(flat_cem)}")
    print(f"Non-unique norm CEM TOTAL: {len(flat_norm_cem_total)}/{len(flat_cem)}")

    print(f"Unique norm CEM SUBS_ID: {len(sorted_norm_cem_sub)}/{len(sorted_cem)}")
    print(f"Unique norm CEM PPI_ID: {len(sorted_norm_cem_ppi)}/{len(sorted_cem)}")
    print(f"Unique norm CEM TOTAL: {len(sorted_norm_cem_total)}/{len(sorted_cem)}")

    # Skewness & kurtosis
    print()
    print("PRO:", get_longtail_stats([x[1] for x in sorted_pro]))
    print("PRO(norm):", get_longtail_stats([x[1] for x in sorted_norm_pro]))
    print("CEM:", get_longtail_stats([x[1] for x in sorted_cem]))
    print("CEM SUBS_ID(norm):", get_longtail_stats([x[1] for x in sorted_norm_cem_sub]))
    print("CEM PPI_ID(norm):", get_longtail_stats([x[1] for x in sorted_norm_cem_ppi]))
    print("CEM TOTAL(norm):", get_longtail_stats([x[1] for x in sorted_norm_cem_total]))


# Sentence tagging for RE data augmentation


def tag_sentence(sentence, rel):
    """
    Insert <substance> and <property> tags into a sentence
    according to two entity spans: rel["entity1"], rel["entity2"].
    """
    a1, a2 = rel["entity1"], rel["entity2"]
    first = min([a1, a2], key=lambda x: x["start"])
    second = max([a1, a2], key=lambda x: x["start"])

    w1 = sentence[first["start"] : first["end"]].strip()
    w2 = sentence[second["start"] : second["end"]].strip()

    if first["entity_group"] == "CEM":
        stag1, etag1 = " <substance> ", " </substance> "
        stag2, etag2 = " <property> ", " </property> "
    else:
        stag1, etag1 = " <property> ", " </property> "
        stag2, etag2 = " <substance> ", " </substance> "

    new_sent = (
        sentence[: first["start"]].strip()
        + stag1
        + w1
        + etag1
        + sentence[first["end"] : second["start"]].strip()
        + stag2
        + w2
        + etag2
        + sentence[second["end"] :].strip()
    )

    return new_sent


# Main entry placeholder

if __name__ == "__main__":
    print("chem_utils.py loaded. Add your experiment script here.")
