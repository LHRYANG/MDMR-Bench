import torch
from typing import List, Optional
from PIL import Image
from diffusers import Flux2Pipeline

from src.inference.base import BaseInferencer


class Flux2Inferencer(BaseInferencer):
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu", hf_token: Optional[str] = None):
        super().__init__(device, hf_token)

    def load_model(self, model_name: str = "black-forest-labs/FLUX.2-dev"):
        print("Loading Flux.2-dev with single GPU and CPU offload...")
        
        try:
            pipe = Flux2Pipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                token=self.hf_token,
                use_safetensors=True
            )
            
            try:
                pipe.enable_model_cpu_offload()
                print("Enabled model CPU offload")
            except Exception as e:
                print(f"Model CPU offload not available, using sequential offload: {e}")
                pipe.enable_sequential_cpu_offload()
                print("Enabled sequential CPU offload")
                
        except Exception as e:
            print(f"Failed to load with offload, falling back to basic device map: {e}")
            pipe = Flux2Pipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                token=self.hf_token,
                device_map="auto",
                use_safetensors=True
            )
        
        try:
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
        except Exception:
            pass
            
        self.model = pipe
        return self.model
    
    def generate_image(self, prompt: str, reference_images: List[Image.Image], output_width: Optional[int] = None, output_height: Optional[int] = None) -> Image.Image:
        inputs = {
            "prompt": prompt,
            "num_inference_steps": 50,
            "guidance_scale": 4.0
        }
        if output_width is not None:
            inputs["width"] = max(16, output_width // 16 * 16)
        if output_height is not None:
            inputs["height"] = max(16, output_height // 16 * 16)
        if len(reference_images) > 0:
            inputs["image"] = reference_images
        output = self.model(**inputs)
        return output.images[0]
