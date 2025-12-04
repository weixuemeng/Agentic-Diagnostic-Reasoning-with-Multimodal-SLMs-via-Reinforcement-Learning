import gradio as gr
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import PeftModel

# -------------------------------
# Load processor + model + LoRA
# -------------------------------
processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    device_map="auto",
    dtype=torch.bfloat16
)

model = PeftModel.from_pretrained(model, "./output/checkpoint-400")
model.eval()


# -------------------------------
# Inference Function
# -------------------------------
def analyze_xray(image, prompt):
    if image is None:
        return "Please upload an image."

    # Chat structure
    messages = [
        {"role": "system", "content": "You are an expert radiologist."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]
        }
    ]

    # Apply Qwen2-VL template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Prepare inputs
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    ).to("cuda")

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200
        )

    raw_output = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    # -------------------------------
    # CLEAN OUTPUT HERE
    # Keep only assistant response
    # -------------------------------
    if "assistant" in raw_output:
        clean_output = raw_output.split("assistant", 1)[-1].strip()
    else:
        clean_output = raw_output.strip()

    return clean_output


# -------------------------------
# Gradio UI
# -------------------------------
with gr.Blocks() as demo:
    gr.Markdown("### Qwen2-VL Radiology Impression Generator (LoRA-finetuned)")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Chest X-ray")
        prompt_input = gr.Textbox(
            label="Prompt",
            value="Analyze this chest X-ray and provide a concise clinical impression."
        )

    output = gr.Textbox(label="Generated Impression")

    run_btn = gr.Button("Generate Report")
    run_btn.click(analyze_xray, inputs=[image_input, prompt_input], outputs=output)

demo.launch()