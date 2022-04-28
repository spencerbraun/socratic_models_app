import json
from PIL import Image

import requests
from transformers import CLIPProcessor, CLIPModel

from embeddings import logger

with open("hf_api.key") as f:
    HF_TOKEN = f.read().strip()


class HuggingFaceHosted:
    def __init__(self, model_id, api_token, verbose=False):
        self.model_id = model_id
        self.api_token = api_token
        self.verbose = verbose

    def query(self, data):
        headers = {"Authorization": f"Bearer {self.api_token}"}
        API_URL = f"https://api-inference.huggingface.co/models/{self.model_id}"
        response = requests.request("POST", API_URL, headers=headers, data=data)
        return json.loads(response.content.decode("utf-8"))

    def fill_mask(self, text):
        data = json.dumps({"inputs": text})
        return self.query(data)

    def text_generation(self, text, **parameters):
        payload = {
            "inputs": text,
            "parameters": parameters,
        }
        if self.verbose:
            logger.info(payload)
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
    def __init__(self, model_id="openai/clip-vit-large-patch14"):
        self.model_id = model_id
        self.model = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def get_image_emb(self, image):
        if isinstance(image, str):
            image = Image.open(image)
        image_inputs = self.processor(images=image, return_tensors="pt", padding=True)
        out = self.model.get_image_features(**image_inputs)

        return out.detach().numpy()

    def get_text_emb(self, text):
        text_inputs = self.processor(text=text, return_tensors="pt", padding=True)
        out = self.model.get_text_features(**text_inputs)

        return out.detach().numpy()

    def __repr__(self):
        return f"CLIP Local <{self.model_id}>"


class GPTJ(HuggingFaceHosted):
    def __init__(
        self, model_id="EleutherAI/gpt-j-6B", api_token=HF_TOKEN, verbose=False
    ):
        super().__init__(model_id, api_token, verbose=verbose)

    def __call__(self, text, **parameters):
        return self.text_generation(text, **parameters)

    def __repr__(self):
        return f"GPTJ Hosted <{self.model_id}>"


class MaskEncoder(HuggingFaceHosted):
    def __init__(self, model_id="roberta-large", api_token=HF_TOKEN, verbose=False):
        super().__init__(model_id, api_token, verbose=verbose)

    def __call__(self, text):
        return self.fill_mask(text)

    def __repr__(self):
        return f"MaskEncoder Hosted <{self.model_id}>"


class T2T(HuggingFaceHosted):
    def __init__(self, model_id="bigscience/T0pp", api_token=HF_TOKEN, verbose=False):
        super().__init__(model_id, api_token, verbose=verbose)

    def __call__(self, text, **parameters):
        return self.text_generation(text, **parameters)

    def __repr__(self):
        return f"T2T Hosted <{self.model_id}>"
