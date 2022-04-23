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

    def predict(self, text):
        payload = {"text": text}
        return self.query(payload, self.model_id, self.api_token)

    def query(self, data):
        headers = {"Authorization": f"Bearer {self.api_token}"}
        API_URL = f"https://api-inference.huggingface.co/models/{self.model_id}"
        response = requests.request("POST", API_URL, headers=headers, data=data)
        return response.json()


class VLM(HuggingFaceHosted):
    """
    Inference class for CLIP model hosted on huggingface inference API
    """

    def __init__(self, model_id, api_token):
        super().__init__(model_id, api_token)
        self.model_id = model_id
        self.api_token = api_token


class CLIP:
    def __init__(self, model_id):
        self.model_id = model_id
        self.model = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
