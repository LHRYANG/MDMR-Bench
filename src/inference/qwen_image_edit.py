import torch
from typing import List, Optional
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

from src.inference.base import BaseInferencer


class QwenImageEditInferencer(BaseInferencer):
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu", hf_token: Optional[str] = None):
        super().__init__(device, hf_token)

    def load_model(self, model_name: str = "Qwen/Qwen-Image-Edit-2511"):
        print(f"Loading Qwen-Image-Edit with single GPU and CPU offload...")
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            token=self.hf_token,
        )
        
        try:
            pipe.enable_model_cpu_offload()
            print("Enabled model CPU offload")
        except Exception as e:
            print(f"Model CPU offload not available, using sequential offload: {e}")
            pipe.enable_sequential_cpu_offload()
            print("Enabled sequential CPU offload")
            
        self.model = pipe
        return self.model
    
    def generate_image(self, prompt: str, reference_images: List[Image.Image], output_width: Optional[int] = None, output_height: Optional[int] = None) -> Image.Image:
        inputs = {
            "image": reference_images,
            "prompt": prompt,
            "negative_prompt": "",
            "num_inference_steps": 50,
            "true_cfg_scale": 4.0
        }
        if output_width is not None:
            inputs["width"] = max(16, output_width // 16 * 16)
        if output_height is not None:
            inputs["height"] = max(16, output_height // 16 * 16)
        output = self.model(**inputs)
        return output.images[0]
