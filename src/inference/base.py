import os
import torch
from typing import List, Optional
from PIL import Image


class BaseInferencer:
    is_api_based = False

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu", hf_token: Optional[str] = None):
        self.device = device
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.model = None

    def load_model(self, model_name: str):
        raise NotImplementedError("Subclasses must implement load_model method")

    def generate_image(
        self,
        prompt: str,
        reference_images: List[Image.Image],
        output_width: Optional[int] = None,
        output_height: Optional[int] = None,
    ) -> Image.Image:
        raise NotImplementedError("Subclasses must implement generate_image method")
