import torch
from peft import PeftModel
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

# 1. Load processor
processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# 2. Load model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    device_map="auto",
    dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(model, "./output/checkpoint-400")
model.eval()

# 3. Load image
image = Image.open("/home/user/workspace/Agentic-Diagnostic-Reasoning-with-Multimodal-SLMs-via-Reinforcement-Learning/Data/Reports/files/p10/p10003019/s55931751/2cd42271-f25135f4-17a199ca-31015e49-c2eb87cb.jpg").convert("RGB")

# 4. Build a chat message (same as training format)
messages = [
    {"role": "system", "content": "You are an expert radiologist."},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Analyze this chest X-ray and provide an impression."},
        ]
    }
]

# 5. Apply chat template (CRITICAL for Qwen2-VL)
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 6. Process image + text together
inputs = processor(
    text=[text],
    images=[image],
    return_tensors="pt"
).to("cuda")

# 7. Generate output
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=200
    )

# 8. Decode
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n=== MODEL OUTPUT ===\n")
print(output_text)
