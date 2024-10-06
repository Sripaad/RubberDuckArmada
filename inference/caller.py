
import os
import sys
import requests
import json
import base64
import logging
from typing import Dict, Any
import store
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from predict import infer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
API_URL = "https://api.mistral.ai/v1/chat/completions"
API_KEY = "cYc1BU2UFIsijNh1Ed7LXicglxctPbOk"  # Consider using environment variables for sensitive information
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# Model configuration
MODEL_NAME = "ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1"
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

def load_model_and_tokenizer():
    logger.info("Loading model and tokenizer")
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_payload(user_text: str, image_paths: list) -> Dict[str, Any]:
    content = [{"type": "text", "text": user_text}]
    for path in image_paths:
        base64_image = encode_image(path)
        content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"})
    
    return {
        "model": "pixtral-12b-2409",
        "messages": [{"role": "user", "content": content}],
        "temperature": 1.0
    }

def call_mistral_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Calling Mistral API")
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    
    if response.ok:
        logger.info("API call successful")
        return response.json()
    logger.error(f"API call failed: {response.status_code} - {response.text}")
    return None

def infer_from(image_path: str):
    logger.info(f"Inferring from image: {image_path}")
    infer(image_path=image_path)

def create_prognosis_report(image_path: str) -> str:
    logger.info(f"Creating prognosis report for image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    prompt = store.prognosis_prompt
    
    msgs = [{'role': 'user', 'content': [prompt, image]}]
    
    logger.info("Generating prognosis report")
    res = model.chat(image=image, msgs=msgs, tokenizer=tokenizer, sampling=True, temperature=1.0, stream=True, max_new_tokens=2048)

    generated_text = ""
    for new_text in res:
        generated_text += new_text
        print(new_text, end='', flush=True)
    
    logger.info("\nPrognosis report generated")
    return generated_text

def analyser_api_call(image_path: str) -> Dict[str, Any]:
    logger.info(f"Starting analysis for image: {image_path}")
    infer_from(image_path)
    prognosis = create_prognosis_report(image_path)
    
    prompt = store.analysis_prompt.replace("$prognosis", prognosis)

    payload = create_payload(prompt, [image_path, "output.jpg", "overlay_output.jpg"])
    return call_mistral_api(payload)

# if __name__ == "__main__":
#     result = analyser_api_call("image.png")
#     if result:
#         logger.info("Analysis completed successfully")
#         print(result)
#     else:
#         logger.error("Analysis failed")