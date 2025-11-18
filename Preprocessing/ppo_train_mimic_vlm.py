import os
import torch
from datasets import Dataset
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from mimic_dataset_vlm import MIMICImpressionDataset
from PIL import Image
from extract_label import chexpert_lookup, LABEL_COLS
import re
import torch.nn as nn

path = "Qwen/Qwen2-VL-7B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    path,
    trust_remote_code=True,          # IMPORTANT for Qwen2-VL
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)

class SimpleValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.v = nn.Linear(hidden_size, 1)
    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, hidden)
        # use last token hidden as a simple pooled representation
        last = hidden_states[:, -1, :]       # (batch, hidden)
        return self.v(last).squeeze(-1)      # (batch,)

# attach it
hidden_size = model.config.hidden_size
model.value_head = SimpleValueHead(hidden_size)

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


def extract_labels_chexpert_style(text):
    txt = text.lower()
    result = {k:0 for k in CHEXPERT_LEXICON}

    for label, terms in CHEXPERT_LEXICON.items():
        for t in terms:
            # Negation + Uncertainty + Finding
            pattern = rf"(no|without|absent|negative for)?\s*(possible|likely|may represent|cannot exclude)?\s*({t})"

            for m in re.finditer(pattern, txt):
                neg, unc, _ = m.groups()

                if neg:
                    result[label] = 0
                elif unc:
                    result[label] = 0.5
                else:
                    result[label] = 1
    return result

# ---- Reward function ----

from sklearn.metrics import f1_score

def clinical_reward(true_dict, pred_dict):
    t = [true_dict[k] for k in CHEXPERT_LEXICON]
    p = [pred_dict[k] for k in CHEXPERT_LEXICON]
    return f1_score(t, p, average="macro", zero_division=0)

def format_reward(text):
    score = 0
    t = text.lower()
    if t.startswith("impression"):
        score += 0.3
    if len(text.split(".")) < 3:
        score -= 0.1
    if len(text) > 800:
        score -= 0.2
    return score

def hallucination_penalty(true_dict, pred_dict):
    penalty = 0
    for k in CHEXPERT_LEXICON:
        if pred_dict[k] == 1 and true_dict[k] == 0:
            penalty -= 0.3
    return penalty

def compute_full_reward(gen_text, true_labels):
    pred_labels = extract_labels_chexpert_style(gen_text)
    R1 = clinical_reward(true_labels, pred_labels)
    R2 = format_reward(gen_text)
    R3 = hallucination_penalty(true_labels, pred_labels)
    return R1 + R2 + R3

# ---- PPO Training Loop ----
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct" # base model 
CSV_PATH = "mimic_impression_subset.csv"
DATA_ROOT = "mimic_subset" # change this to 
STEPS = 100  # start small for testing

# ---- Dataset ----
mimic_ds = MIMICImpressionDataset(CSV_PATH, DATA_ROOT)
#hf_ds = Dataset.from_dict({
#    "study_id": [d["study_id"] for d in mimic_ds],
#    "prompt":   [d["prompt"] for d in mimic_ds],
#    "reference":[d["reference"] for d in mimic_ds],
#    "image":    [d["image"] for d in mimic_ds],
#})

# ---- Model + Processor ----
#processor = AutoProcessor.from_pretrained(MODEL_NAME)
#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#tokenizer.pad_token = tokenizer.eos_token

#model = AutoModelForCausalLMWithValueHead.from_pretrained(
#    MODEL_NAME,
#    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
#    device_map="auto"
#)

# ---- PPO Config ----
config = PPOConfig(
    batch_size=8,
    mini_batch_size=4,
    learning_rate=1e-6,
    gradient_accumulation_steps=1,   # <- changed from 4 to 1
    target_kl=0.1,
    ppo_epochs=4
)

trainer = PPOTrainer(config, model, tokenizer, dataset=mimic_ds)

# ---- Reward Function ----
# def compute_reward(reference, generated):
#     ref, gen = reference.lower(), generated.lower()
#     overlap = len(set(ref.split()) & set(gen.split())) / max(len(set(ref.split())), 1)
#     format_bonus = 1.0 if gen.strip().startswith("impression") else 0.0
#     return overlap + format_bonus

# ---- PPO Loop ----
for step, batch in enumerate(trainer.dataset):

    img = batch["image"]
    prompt = batch["prompt"]
    sid = batch["study_id"]

    true_labels = chexpert_lookup[int(sid)]

    inputs = processor(images=img, text=prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=128)

    gen_text = tokenizer.decode(output[0], skip_special_tokens=True)

    reward = compute_full_reward(gen_text, true_labels)

    trainer.step([gen_text], torch.tensor([reward], device=model.device))

    if step % 10 == 0:
        print(f"[Step {step}] reward={reward:.3f}")


    if step >= STEPS:
        break

trainer.save_pretrained("mimic_vlm_ppo_model")
print("PPO fine-tuning complete!")