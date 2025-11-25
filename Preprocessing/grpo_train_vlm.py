import os
from typing import List, Optional, Dict, Any
import torch
from datasets import Dataset, DatasetDict
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
from mimic_dataset_vlm import MIMICImpressionDataset
from PIL import Image
from extract_label import chexpert_lookup, LABEL_COLS
import re
import io
from sklearn.metrics import f1_score

# ---------------- dataset ------------------
CSV_PATH = "mimic_jpg_impression_subset.csv"
IMAGE_ROOT = "../Data/Reports"
full = MIMICImpressionDataset(CSV_PATH, IMAGE_ROOT)

records = []
for i in range(len(full)):
    item = full[i]
    item["labels"] = chexpert_lookup[int(item["study_id"])]
    records.append(item)

train = Dataset.from_list([r for r in records if r["split"] == "train"])
eval  = Dataset.from_list([r for r in records if r["split"] == "eval"])
test  = Dataset.from_list([r for r in records if r["split"] == "test"])

ds = DatasetDict({
    "train": train,
    "eval": eval,
    "test": test
})
print(ds)

# ---------------- reward ------------------
# extract labels using CheXpert-style rules
CHEXPERT_LEXICON = {
    "Atelectasis": ["atelectasis", "atelectatic", "collapse"],
    "Cardiomegaly": ["cardiomegaly", "enlarged heart"],
    "Consolidation": ["consolidation", "airspace disease"],
    "Edema": ["edema", "pulmonary edema", "interstitial edema"],
    "Enlarged Cardiomediastinum": ["widened mediastinum"],
    "Fracture": ["fracture"],
    "Lung Lesion": ["mass", "nodule", "lesion"],
    "Lung Opacity": ["opacity", "opacities"],
    "Pleural Effusion": ["effusion", "fluid"],
    "Pleural Other": ["pleural thickening"],
    "Pneumonia": ["pneumonia", "infectious infiltrate"],
    "Pneumothorax": ["pneumothorax", "collapsed lung"],
    "Support Devices": ["endotracheal", "line", "catheter"],
    "No Finding": ["normal", "no acute disease", "clear lungs"]
}

NEG_PHRASES = ["no", "without", "absent", "negative for"]
UNCERTAIN_PHRASES = ["possible", "likely", "may represent", "cannot exclude"]

def extract_labels_from_text(text: str):
    """
    extract the label from the predicted text
    """
    txt = text.lower()
    result = {k: 0.0 for k in CHEXPERT_LEXICON}

    for label, terms in CHEXPERT_LEXICON.items():
        for t in terms:
            # (optional negation) (optional uncertainty) (term)
            pattern = rf"(no|without|absent|negative for)?\s*(possible|likely|may represent|cannot exclude)?\s*({re.escape(t)})"
            for m in re.finditer(pattern, txt):
                neg, unc, _ = m.groups()
                if neg:
                    result[label] = 0.0
                elif unc:
                    result[label] = max(result[label], 0.5)
                else:
                    result[label] = 1.0
    return result
                 
def format_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Simple formatting/content prior:
    +0.3 if starts with 'impression'
    -0.1 if fewer than 3 sentences
    -0.2 if too long (> 800 chars)
    """
    rewards: List[float] = []
    for text in completions:
        t = text.strip().lower()
        r = 0.0
        if t.startswith("impression"):
            r += 0.3
        if len(text.split(".")) < 3:
            r -= 0.1
        if len(text) > 800:
            r -= 0.2
        rewards.append(r)
    return rewards

def chexpert_f1_reward(
    completions: List[str],
    labels: List[Dict[str, int]],
    **kwargs,
) -> List[Optional[float]]:
    """
    Reward based on agreement between generated text and CheXpert labels.
    Uses macro-F1 over LABEL_COLS.
    """
    rewards: List[Optional[float]] = []

    for text, true_dict in zip(completions, labels):
        try:
            pred_soft = extract_labels_from_text(text)
            # Convert soft predictions → binary (>=0.5 → 1, else 0)
            y_true = [int(true_dict.get(lbl, 0)) for lbl in LABEL_COLS]
            y_pred = [1 if pred_soft.get(lbl, 0.0) >= 0.5 else 0 for lbl in LABEL_COLS]

            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            rewards.append(float(f1))
        except Exception as e:
            print(f"[WARN] chexpert_f1_reward failed: {e}")
            rewards.append(None)

    return rewards

# ---------------- model ------------------
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct" # base model 
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
     use_fast=True,
    padding_side="left",
)

SYSTEM_PROMPT = (
    "You are an expert radiologist. Given the chest X-ray image, "
    "provide ONLY the Impression. Be concise and medically accurate."
)

def add_chat_prompt(example):
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": example["prompt"]},
            ],
        },
    ]

    example["prompt"] = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    return example

ds = ds.map(add_chat_prompt)

# checking 
def bytes_to_pil(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    For sanity: ensure 'image' is raw PNG bytes.
    (Your MIMICImpressionDataset already returns PNG bytes, so this is a no-op
    except for checking / enforcing type.)
    """
    if isinstance(example["image"], bytes):
        # leave as-is; processor will handle `image` as bytes->PIL internally
        return example
    # if something weird slipped in, convert here (optional)
    return example

ds = ds.map(bytes_to_pil)

model = AutoModelForCausalLM.from_pretrained(
     MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# ------------ GRPO Config Trainer -------------
training_args = GRPOConfig(
    output_dir="qwen2_vl_mimic_grpo",
    learning_rate=1e-5,
    remove_unused_columns=False, # to access the solution column in accuracy_reward
    num_train_epochs=1,
    bf16=True,

    per_device_train_batch_size=2,
    num_generations=2, # default: 8
    max_prompt_length=2048,

    # Parameters related to reporting and saving
    report_to=[],
    logging_steps=10,
    save_strategy="steps",
    save_steps=10,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=processor,
    reward_funcs=[format_reward, chexpert_f1_reward],
    args=training_args,
    train_dataset=ds["train"],
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model("qwen2_vl_mimic_grpo_final")
    print("GRPO fine-tuning complete!")


