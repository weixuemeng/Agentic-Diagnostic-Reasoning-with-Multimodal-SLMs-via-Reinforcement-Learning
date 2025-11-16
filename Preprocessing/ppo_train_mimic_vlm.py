import os
import torch
from datasets import Dataset
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
from transformers import AutoProcessor, AutoTokenizer
from dataset_mimic_vlm import MIMICImpressionDataset
from PIL import Image

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct" # base model 
CSV_PATH = "mimic_impression_subset.csv"
DATA_ROOT = "mimic_subset" # change this to 
STEPS = 100  # start small for testing

# ---- Dataset ----
mimic_ds = MIMICImpressionDataset(CSV_PATH, DATA_ROOT)
hf_ds = Dataset.from_dict({
    "study_id": [d["study_id"] for d in mimic_ds],
    "prompt":   [d["prompt"] for d in mimic_ds],
    "reference":[d["reference"] for d in mimic_ds],
    "image":    [d["image"] for d in mimic_ds],
})

# ---- Model + Processor ----
processor = AutoProcessor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# ---- PPO Config ----
config = PPOConfig(
    batch_size=8,
    mini_batch_size=4,
    learning_rate=1e-6,
    gradient_accumulation_steps=4,
    target_kl=0.1,
    ppo_epochs=4
)

trainer = PPOTrainer(config, model, tokenizer, dataset=hf_ds)

# ---- Reward Function ----
def compute_reward(reference, generated):
    ref, gen = reference.lower(), generated.lower()
    overlap = len(set(ref.split()) & set(gen.split())) / max(len(set(ref.split())), 1)
    format_bonus = 1.0 if gen.strip().startswith("impression") else 0.0
    return overlap + format_bonus



# ---- PPO Loop ----
for step, batch in enumerate(trainer.dataset):
    prompt = batch["prompt"] # 8 
    image = batch["image"]
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, max_new_tokens=128, temperature=0.7)
    gen_text = tokenizer.decode(output[0], skip_special_tokens=True) # get the generated impression from the base model

    reward = []
    for ref, gen in zip(batch["reference"], [gen_text]*len(batch["reference"])):
        reward.append(compute_reward(ref, gen))

    reward = torch.tensor([compute_reward(batch["reference"], gen_text)], device=model.device)
    trainer.step([gen_text], reward)

    if step % 10 == 0:
        print(f"[Step {step}] Reward = {reward.item():.3f}")

    if step >= STEPS:
        break

trainer.save_pretrained("mimic_vlm_ppo_model")
print("PPO fine-tuning complete!")
