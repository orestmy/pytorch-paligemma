# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading model...")
start_time = time.time()
# Load model directly
processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
model = AutoModelForImageTextToText.from_pretrained("google/gemma-3-4b-it")
print("Loading to GPU")
model = model.to(device)
load_time = time.time() - start_time
print(f"Model loaded in {load_time:.2f} seconds")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

print("Running inference...")
start_time = time.time()
outputs = model.generate(**inputs, max_new_tokens=100)
inference_time = time.time() - start_time
print(f"Inference completed in {inference_time:.2f} seconds")

response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])
print(f"Response: {response}")