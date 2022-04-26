import json
import os
import yaml
from PIL import Image

import requests
import torch
import transformers
from transformers import CLIPProcessor, CLIPModel

with open("hf_api.key") as f:
    HF_TOKEN = f.read().strip()


class HuggingFaceHosted:
    def __init__(self, model_id, api_token):
        self.model_id = model_id
        self.api_token = api_token

    def query(self, data):
        headers = {"Authorization": f"Bearer {self.api_token}"}
        API_URL = f"https://api-inference.huggingface.co/models/{self.model_id}"
        response = requests.request("POST", API_URL, headers=headers, data=data)
        return json.loads(response.content.decode("utf-8"))

    def text_generation(self, text):
        payload = {"inputs": text}
        data = json.dumps(payload)
        return self.query(data)

    def summarization(self, text, do_sample=False):
        data = json.dumps({"inputs": text, "parameters": {"do_sample": do_sample}})
        return self.query(data)

    def question_answering(self, question, context):
        data = json.dumps(
            {
                "inputs": {
                    "question": question,
                    "context": context,
                }
            }
        )
        return self.query(data)


class CLIP:
    def __init__(self, model_id):
        self.model_id = model_id
        self.model = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def from_file(self, image_path):
        image = Image.open(image_path)
        return self.from_image(image)

    def from_image(self, image):
        image_inputs = self.processor(images=image, return_tensors="pt", padding=True)
        out = self.model.get_image_features(**image_inputs)

        return out.detach().numpy()


class GPTJ(HuggingFaceHosted):
    def __init__(self, model_id="EleutherAI/gpt-j-6B", api_token=HF_TOKEN):
        super().__init__(model_id, api_token)

    def __call__(self, text):
        return self.text_generation(text)
