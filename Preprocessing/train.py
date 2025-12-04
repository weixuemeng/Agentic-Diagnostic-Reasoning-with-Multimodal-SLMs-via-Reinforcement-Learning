import sys
log_file = open("log.txt", "w")
sys.stdout = log_file
sys.stderr = log_file


import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from datasets import Dataset, DatasetDict
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoModelForVision2Seq, Qwen2VLProcessor, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
import re
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
import math
import io
from typing import List, Dict, Any
from extract_label import chexpert_lookup, LABEL_COLS
from peft import LoraConfig, get_peft_model
import warnings
import matplotlib.pyplot as plt
from transformers import TrainerCallback
import json
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Config:
    MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
    CSV_PATH = "/home/user/workspace/Agentic-Diagnostic-Reasoning-with-Multimodal-SLMs-via-Reinforcement-Learning/Preprocessing/mimic_jpg_impression_subset.csv"
    IMAGE_ROOT = "/home/user/workspace/Agentic-Diagnostic-Reasoning-with-Multimodal-SLMs-via-Reinforcement-Learning/Data/Reports"
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 2
    NUM_EPOCHS = 10
    OUTPUT_DIR = "qwen2_vl_mimic_grpo"
    MAX_LENGTH = 2048
    MAX_PROMPT_LENGTH = 1024
    LOGGING_STEPS = 10
    SAVE_STEPS = 50
    EVAL_STEPS = 50
    EVAL_STRATEGY = "steps"
    SAVE_STRATEGY = "steps"
    METRIC_FOR_BEST_MODEL = "eval_loss"
    LOAD_BEST_MODEL_AT_END = True
    WARMUP_STEPS = 0
    DATASET_KWARGS = {"skip_prepare_data": True}
    REMOVE_UNUSED_COLUMNS = False
    MAX_SEQ_LEN = 128
    GRADIENT_CHECKPOINTING = True
    MAX_GRAD_NORM = 1.0

config = Config()

# Custom callback to track metrics
class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        self.eval_steps = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
                self.steps.append(state.global_step)
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])
                self.eval_steps.append(state.global_step)

metrics_callback = MetricsCallback()

def build_clinical_context_prompt(sample):
    """Build prompt with clinical context"""
    context_parts = []
    
    if sample.get("indication") and sample["indication"].strip():
        indication = sample["indication"].strip()
        if len(indication) > 0:
            context_parts.append(f"INDICATION: {indication}")
    
    if sample.get("comparision") and sample["comparision"].strip():
        comparison = sample["comparision"].strip()
        if len(comparison) > 0:
            context_parts.append(f"COMPARISON: {comparison}")
    
    if sample.get("technique") and sample["technique"].strip():
        technique = sample["technique"].strip()
        context_parts.append(f"TECHNIQUE: {technique}")
    
    if sample.get("examination") and sample["examination"].strip():
        examination = sample["examination"].strip()
        context_parts.append(f"EXAMINATION: {examination}")
    
    if context_parts:
        context = "\n".join(context_parts)
        return f"Clinical context:\n{context}\n\nAnalyze this chest X-ray and provide a clinical impression."
    
    return "Analyze this chest X-ray and provide a clinical impression."

SYSTEM_PROMPT = (
    "A conversation between expert radiologist and the patient. Given the chest X-ray image, "
    "first thinks about the reasoning process in the mind and then provides the patient with the impression. "
    "Be concise and medically accurate."
)

def format_data(sample):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": build_clinical_context_prompt(sample),
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["reference"]}],
        },
    ]

from mimic_dataset_vlm import MIMICImpressionDataset

CSV_PATH = "/home/user/workspace/Agentic-Diagnostic-Reasoning-with-Multimodal-SLMs-via-Reinforcement-Learning/Preprocessing/mimic_jpg_impression_subset.csv"
IMAGE_ROOT = "/home/user/workspace/Agentic-Diagnostic-Reasoning-with-Multimodal-SLMs-via-Reinforcement-Learning/Data/Reports"

full_dataset = MIMICImpressionDataset(CSV_PATH, IMAGE_ROOT)
print(f"Loaded dataset with {len(full_dataset)} samples")

records = []
success_count = 0

for i in range(min(len(full_dataset), 100)):
    try:
        item = full_dataset[i]
        sid = item.get("study_id")
        
        if isinstance(item["image"], bytes):
            image_pil = Image.open(io.BytesIO(item["image"]))
            item["image"] = image_pil
        
        item["reference"] = str(item.get("reference", ""))
        item["indication"] = str(item.get('indication', ''))
        item["examination"] = str(item.get('examination', ''))
        item["technique"] = str(item.get('technique', ''))
        item["comparision"] = str(item.get('comparision', ''))
        
        if sid not in chexpert_lookup:
            continue
        
        item["label"] = chexpert_lookup[sid]
        
        sp = item.get("split", "")
        if isinstance(sp, str) and sp.strip().lower() == "validate":
            item["split"] = "eval"
        
        records.append(item)
        success_count += 1
    except Exception as e:
        print(f"Error processing item {i}: {e}")
        continue

print(f"Successfully processed {success_count} records")

def create_splits(records):
    train_target = "train".strip().lower()
    train_rows = [r for r in records if str(r.get("split", "")).strip().lower() == train_target]
    
    eval_target = "eval".strip().lower()
    eval_rows = [r for r in records if str(r.get("split", "")).strip().lower() == eval_target]
    
    test_target = "test".strip().lower()
    test_rows = [r for r in records if str(r.get("split", "")).strip().lower() == test_target]
    
    return {
        "train": Dataset.from_list(train_rows),
        "eval": Dataset.from_list(eval_rows),
        "test": Dataset.from_list(test_rows)
    }

ds_dict = create_splits(records)
ds = DatasetDict(ds_dict)
print(f"Dataset splits - Train: {len(ds['train'])}, Eval: {len(ds['eval'])}, Test: {len(ds['test'])}")

train_dataset = ds["train"]
eval_dataset = ds["eval"]
test_dataset = ds["test"]

print(len(train_dataset))
print("-"*30)
print(train_dataset)
print("-"*30)
print(train_dataset[0])
print("-"*30)

train_dataset = [format_data(sample) for sample in train_dataset]
eval_dataset = [format_data(sample) for sample in eval_dataset]
test_dataset = [format_data(sample) for sample in test_dataset]

print(len(train_dataset))
print("-"*30)
print(train_dataset[0])
print("-"*30)
print(len(test_dataset))
print("-"*30)
print(test_dataset[0])

sample_data = test_dataset[0]
sample_question = test_dataset[0][1]["content"][1]["text"]
sample_answer = test_dataset[0][2]["content"][0]["text"]
sample_image = test_dataset[0][1]["content"][0]["image"]

print(sample_question)
print(sample_answer)

print("Loading model and processor...")

if torch.cuda.is_available():
    print("Using GPU for model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
else:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.float32,
    )

processor = Qwen2VLProcessor.from_pretrained(
    config.MODEL_NAME,
)

if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

processor.tokenizer.padding_side = "right"

def text_generator(sample):
    text = processor.apply_chat_template(
        sample[0:2],
        tokenize=False,
        add_generation_prompt=True
    )
    
    image_inputs = sample[1]["content"][0]["image"]
    
    inputs = processor(
        text=[text],
        images=[image_inputs],
        return_tensors="pt",
        padding=True,
    ).to(device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=config.MAX_SEQ_LEN)
    output_text = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True
    )
    
    del inputs
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    actual_answer = sample[2]["content"][0]["text"]
    return output_text[0], actual_answer

print("\n" + "="*50)
print("BEFORE TRAINING - Sample Generation")
print("="*50)
generated_text, actual_answer = text_generator(sample_data)
print(f"Generated Answer: {generated_text}")
print(f"Actual Answer: {actual_answer}")

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

print(f"Before adapter parameters: {model.num_parameters()}")

training_args = SFTConfig(
    output_dir="./output",
    num_train_epochs=config.NUM_EPOCHS,
    per_device_train_batch_size=config.BATCH_SIZE,
    per_device_eval_batch_size=config.BATCH_SIZE,
    gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
    gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
    learning_rate=config.LEARNING_RATE,
    logging_steps=config.LOGGING_STEPS,
    eval_steps=config.EVAL_STEPS,
    eval_strategy=config.EVAL_STRATEGY,
    save_strategy=config.SAVE_STRATEGY,
    save_steps=config.SAVE_STEPS,
    metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
    load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
    max_grad_norm=config.MAX_GRAD_NORM,
    warmup_steps=config.WARMUP_STEPS,
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
    report_to="none",
    optim="adamw_torch",
    fp16=False,
    bf16=torch.cuda.is_available(),
)

def collate_fn(examples):
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    image_inputs = [example[1]["content"][0]["image"] for example in examples]
    
    batch = processor(
        text=texts, 
        images=image_inputs, 
        return_tensors="pt", 
        padding=True
    )
    
    labels = batch["input_ids"].clone()
    pad_id = processor.tokenizer.pad_token_id
    labels[labels == pad_id] = -100
    batch["labels"] = labels
    
    return batch

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
    processing_class=processor.tokenizer,
    callbacks=[metrics_callback],
)

print("-"*30)
print("Initial Evaluation")
metric = trainer.evaluate()
print(metric)
print("-"*30)
print("Training")
trainer.train()
print("-"*30)

# Plot training and evaluation loss
print("\n" + "="*50)
print("PLOTTING TRAINING METRICS")
print("="*50)

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

if metrics_callback.train_losses:
    ax.plot(metrics_callback.steps, metrics_callback.train_losses, 
            label='Training Loss', marker='o', linewidth=2, markersize=4)

if metrics_callback.eval_losses:
    ax.plot(metrics_callback.eval_steps, metrics_callback.eval_losses, 
            label='Evaluation Loss', marker='s', linewidth=2, markersize=4)

ax.set_xlabel('Training Steps', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Training and Evaluation Loss Over Time', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
print("Saved training metrics plot to 'training_metrics.png'")
plt.close()

# Final evaluation on test set
print("\n" + "="*50)
print("FINAL EVALUATION ON TEST SET")
print("="*50)

def evaluate_on_dataset(dataset, dataset_name="Test"):
    """Generate predictions and compute metrics"""
    predictions = []
    references = []
    
    print(f"\nGenerating predictions for {dataset_name} set...")
    for i, sample in enumerate(dataset):
        try:
            generated, reference = text_generator(sample)
            predictions.append(generated)
            references.append(reference)
            
            if i < 3:  # Print first 3 examples
                print(f"\nExample {i+1}:")
                print(f"Generated: {generated[:200]}...")
                print(f"Reference: {reference[:200]}...")
        except Exception as e:
            print(f"Error generating prediction {i}: {e}")
            predictions.append("")
            references.append(sample[2]["content"][0]["text"])
    
    return predictions, references

# Evaluate on test set
test_predictions, test_references = evaluate_on_dataset(test_dataset, "Test")

# Evaluate on eval set for comparison
eval_predictions, eval_references = evaluate_on_dataset(eval_dataset, "Validation")

# Calculate text similarity metrics
from difflib import SequenceMatcher

def calculate_similarity(pred, ref):
    """Calculate similarity ratio between two strings"""
    return SequenceMatcher(None, pred.lower(), ref.lower()).ratio()

def calculate_bleu_score(predictions, references):
    """Simple word-level BLEU approximation"""
    from collections import Counter
    scores = []
    for pred, ref in zip(predictions, references):
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        if len(ref_words) > 0:
            precision = len(pred_words & ref_words) / len(pred_words) if len(pred_words) > 0 else 0
            scores.append(precision)
    return np.mean(scores) if scores else 0

# Calculate metrics
test_similarities = [calculate_similarity(p, r) for p, r in zip(test_predictions, test_references)]
eval_similarities = [calculate_similarity(p, r) for p, r in zip(eval_predictions, eval_references)]

test_bleu = calculate_bleu_score(test_predictions, test_references)
eval_bleu = calculate_bleu_score(eval_predictions, eval_references)

# Length statistics
test_pred_lengths = [len(p.split()) for p in test_predictions]
test_ref_lengths = [len(r.split()) for r in test_references]
eval_pred_lengths = [len(p.split()) for p in eval_predictions]
eval_ref_lengths = [len(r.split()) for r in eval_references]

print("\n" + "="*50)
print("EVALUATION METRICS SUMMARY")
print("="*50)

print("\nTest Set Metrics:")
print(f"  Average Similarity Score: {np.mean(test_similarities):.4f} (±{np.std(test_similarities):.4f})")
print(f"  BLEU Score (Word Overlap): {test_bleu:.4f}")
print(f"  Avg Prediction Length: {np.mean(test_pred_lengths):.1f} words")
print(f"  Avg Reference Length: {np.mean(test_ref_lengths):.1f} words")

print("\nValidation Set Metrics:")
print(f"  Average Similarity Score: {np.mean(eval_similarities):.4f} (±{np.std(eval_similarities):.4f})")
print(f"  BLEU Score (Word Overlap): {eval_bleu:.4f}")
print(f"  Avg Prediction Length: {np.mean(eval_pred_lengths):.1f} words")
print(f"  Avg Reference Length: {np.mean(eval_ref_lengths):.1f} words")

# Plot evaluation metrics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Similarity scores distribution
axes[0, 0].hist(test_similarities, bins=20, alpha=0.7, label='Test', color='blue', edgecolor='black')
axes[0, 0].hist(eval_similarities, bins=20, alpha=0.7, label='Validation', color='orange', edgecolor='black')
axes[0, 0].set_xlabel('Similarity Score', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('Distribution of Similarity Scores', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Length comparison
datasets = ['Test', 'Validation']
pred_lengths = [np.mean(test_pred_lengths), np.mean(eval_pred_lengths)]
ref_lengths = [np.mean(test_ref_lengths), np.mean(eval_ref_lengths)]
x = np.arange(len(datasets))
width = 0.35
axes[0, 1].bar(x - width/2, pred_lengths, width, label='Predictions', color='skyblue', edgecolor='black')
axes[0, 1].bar(x + width/2, ref_lengths, width, label='References', color='lightcoral', edgecolor='black')
axes[0, 1].set_xlabel('Dataset', fontsize=11)
axes[0, 1].set_ylabel('Average Length (words)', fontsize=11)
axes[0, 1].set_title('Average Text Length Comparison', fontsize=12, fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(datasets)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# BLEU scores
axes[1, 0].bar(datasets, [test_bleu, eval_bleu], color=['blue', 'orange'], 
               alpha=0.7, edgecolor='black')
axes[1, 0].set_ylabel('BLEU Score', fontsize=11)
axes[1, 0].set_title('BLEU Scores by Dataset', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Summary metrics
metrics_data = {
    'Metric': ['Avg Similarity', 'BLEU Score', 'Avg Pred Length', 'Avg Ref Length'],
    'Test': [np.mean(test_similarities), test_bleu, np.mean(test_pred_lengths), np.mean(test_ref_lengths)],
    'Validation': [np.mean(eval_similarities), eval_bleu, np.mean(eval_pred_lengths), np.mean(eval_ref_lengths)]
}
axes[1, 1].axis('tight')
axes[1, 1].axis('off')
table_data = [[k, f"{v:.3f}", f"{metrics_data['Validation'][i]:.3f}"] 
              for i, (k, v) in enumerate(zip(metrics_data['Metric'], metrics_data['Test']))]
table = axes[1, 1].table(cellText=table_data, 
                         colLabels=['Metric', 'Test', 'Validation'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.4, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
axes[1, 1].set_title('Metrics Summary Table', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('evaluation_metrics.png', dpi=300, bbox_inches='tight')
print("\nSaved evaluation metrics plot to 'evaluation_metrics.png'")
plt.close()

# Save detailed results to JSON
results = {
    'test_metrics': {
        'avg_similarity': float(np.mean(test_similarities)),
        'std_similarity': float(np.std(test_similarities)),
        'bleu_score': float(test_bleu),
        'avg_pred_length': float(np.mean(test_pred_lengths)),
        'avg_ref_length': float(np.mean(test_ref_lengths)),
    },
    'validation_metrics': {
        'avg_similarity': float(np.mean(eval_similarities)),
        'std_similarity': float(np.std(eval_similarities)),
        'bleu_score': float(eval_bleu),
        'avg_pred_length': float(np.mean(eval_pred_lengths)),
        'avg_ref_length': float(np.mean(eval_ref_lengths)),
    }
}

with open('evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved detailed results to 'evaluation_results.json'")

print("\n" + "="*50)
print("TRAINING AND EVALUATION COMPLETE!")
print("="*50)
print("\nGenerated files:")
print("  - training_metrics.png: Training/validation loss curves")
print("  - evaluation_metrics.png: Comprehensive evaluation visualizations")
print("  - evaluation_results.json: Detailed metrics in JSON format")

print("\nSaving final merged full model...")

from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration

# Load the original base model again
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    config.MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Load the LAST LoRA checkpoint created by SFTTrainer
final_adapter_path = trainer.state.best_model_checkpoint or trainer.state.last_model_checkpoint

print(f"Loading LoRA adapter from: {final_adapter_path}")

lora_model = PeftModel.from_pretrained(base_model, final_adapter_path)

# Merge LoRA weights into the base model
merged_model = lora_model.merge_and_unload()

# Save merged full model
save_path = "./final_merged_model"
merged_model.save_pretrained(save_path)

# Save tokenizer
processor.tokenizer.save_pretrained(save_path)

print(f"Final merged full model saved to: {save_path}")

log_file.close()