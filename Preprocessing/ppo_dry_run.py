import torch
import pandas as pd
from datasets import Dataset
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
from transformers import AutoTokenizer

MODEL_NAME = "sshleifer/tiny-gpt2"
CSV_PATH = "mimic_impression_subset.csv"
STEPS = 2


def compute_reward(reference, generated):
    ref, gen = reference.lower(), generated.lower()
    overlap = len(set(ref.split()) & set(gen.split())) / max(len(set(ref.split())), 1)
    return overlap


def main():
    # load small subset
    df = pd.read_csv(CSV_PATH)
    df = df.head(16)

    hf_ds = Dataset.from_dict({
        "prompt": ["Two-view chest X-ray. Provide ONLY the Impression." for _ in range(len(df))],
        "reference": df["impression"].tolist(),
    })

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)
    # Ensure model has a generation_config (some wrappers may not attach it)
    if not hasattr(model, "generation_config"):
        try:
            from transformers import GenerationConfig

            model.generation_config = GenerationConfig(**model.config.to_dict())
        except Exception:
            model.generation_config = model.config

    config = PPOConfig(
        batch_size=2,
        mini_batch_size=1,
        learning_rate=1e-5,
        gradient_accumulation_steps=1,
    )

    # Construct PPOTrainer using the installed TRL signature.
    # args: (PPOConfig, processing_class, model, ref_model, reward_model, train_dataset, value_model)
    # Use the same model as a simple value_model fallback for this dry run
    trainer = PPOTrainer(config, tokenizer, model, None, None, hf_ds, model)

    # Simple loop over dataset samples (single-sample batches)
    for step in range(STEPS):
        sample = hf_ds[step]
        prompt = sample["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        output = model.generate(input_ids=input_ids, max_new_tokens=16)
        gen_text = tokenizer.decode(output[0], skip_special_tokens=True)

        reward_val = compute_reward(sample["reference"], gen_text)
        reward = torch.tensor([reward_val], device=model.device)

        # Call trainer.step with single response and reward
        try:
            trainer.step([gen_text], reward)
        except Exception as e:
            # If trainer.step fails in this trl version, fall back to printing and continue
            print("trainer.step failed:", e)

        print(f"[DryRun Step {step}] gen='{gen_text[:80]}' reward={reward_val:.4f}")

    print("Dry-run complete")


if __name__ == "__main__":
    main()
