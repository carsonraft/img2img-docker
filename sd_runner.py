import os
import time
from typing import List
import base64
from io import BytesIO

import torch
from PIL import Image

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

MODEL_ID = "stabilityai/stable-diffusion-2-1"
MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"


class Predictor:
    ''' A predictor class that loads the model into memory and runs predictions '''

    def __init__(self, model_id):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def setup(self):
        start_time = time.time()
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_MODEL_ID,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            # safety_checker=safety_checker,
            safety_checker=None,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to(self.device)
        
        # Load img2img pipeline using same components
        self.img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.pipe.vae,
            text_encoder=self.pipe.text_encoder,
            tokenizer=self.pipe.tokenizer,
            unet=self.pipe.unet,
            scheduler=self.pipe.scheduler,
            safety_checker=None,
            feature_extractor=self.pipe.feature_extractor,
        ).to(self.device)

        self.pipe.enable_xformers_memory_efficient_attention()
        self.img2img_pipe.enable_xformers_memory_efficient_attention()
        # self.pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        end_time = time.time()
        print(f"setup time: {end_time - start_time}")

    @torch.inference_mode()
    def predict(self, prompt, negative_prompt, width, height, num_outputs, num_inference_steps, guidance_scale, scheduler, seed, image=None, strength=0.8):
        """Run a single prediction on the model"""
        start_time = time.time()
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        generator = torch.Generator(self.device).manual_seed(seed)
        
        # Route to img2img if image is provided
        if image is not None:
            # Decode base64 image
            if isinstance(image, str):
                image_data = base64.b64decode(image)
                init_image = Image.open(BytesIO(image_data)).convert("RGB")
            else:
                init_image = image
            
            self.img2img_pipe.scheduler = make_scheduler(scheduler, self.img2img_pipe.scheduler.config)
            
            output = self.img2img_pipe(
                prompt=[prompt] * num_outputs if prompt is not None else None,
                negative_prompt=[negative_prompt] * num_outputs
                if negative_prompt is not None
                else None,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                generator=generator,
                num_inference_steps=num_inference_steps,
            )
        else:
            # Use txt2img pipeline
            self.pipe.scheduler = make_scheduler(scheduler, self.pipe.scheduler.config)
            
            output = self.pipe(
                prompt=[prompt] * num_outputs if prompt is not None else None,
                negative_prompt=[negative_prompt] * num_outputs
                if negative_prompt is not None
                else None,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                generator=generator,
                num_inference_steps=num_inference_steps,
            )

        output_paths = []
        for i, sample in enumerate(output.images):
            if output.nsfw_content_detected and output.nsfw_content_detected[i]:
                continue

            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(output_path)

        if len(output_paths) == 0:
            raise Exception(
                "NSFW content detected. Try running it again, or try a different prompt.")
        end_time = time.time()
        print(f"inference took {end_time - start_time} time")
        return output_paths


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]
