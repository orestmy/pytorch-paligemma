from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
import io
import base64
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model

app = Flask(__name__)

model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    global model, processor
    
    model_path = os.getenv("MODEL_PATH", "../paligemma-weights/")
    print(f"Loading models from: {model_path}")
    
    start_time = time.time()
    print("Loading model...")
    
    model, tokenizer = load_hf_model(model_path, device)
    model = model.eval()
    
    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")

def test_inference(prompt: str, image, max_tokens: int = 100, temperature: float = 0.8, 
                  top_p: float = 0.9, do_sample: bool = False):
    global model, processor

    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    for _ in range(max_tokens):
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]

        if do_sample:
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(next_token_logits, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1)
            next_token = torch.gather(probs_idx, -1, next_token)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        next_token = next_token.squeeze(0)
        generated_tokens.append(next_token)

        if next_token.item() == stop_token:
            break

        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )

    generated_tokens = torch.cat(generated_tokens, dim=-1)
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return decoded

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if model is None or processor is None:
        return jsonify({'error': 'Models not loaded'}), 500
    try:
        image_data = request.json['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        question = request.json['question']
        
        print(f"Processing question: {question}")
        start_time = time.time()

        with torch.no_grad():
            response = test_inference(
                prompt=question,
                image=image,
                max_tokens=100,
                temperature=0.8,
                top_p=0.9,
                do_sample=True
            )
            
        inference_time = time.time() - start_time
        print(f"Generated response in {inference_time:.2f} seconds: {response}")

        return jsonify({'response': response})
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

print(f"Using device: {device}")
load_models()
print("Starting Flask server...")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
