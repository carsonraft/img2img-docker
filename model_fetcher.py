'''
RunPod | serverless-ckpt-template | model_fetcher.py

Downloads the model from the URL passed in.
'''

import shutil
import requests
import argparse
from pathlib import Path
from urllib.parse import urlparse

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"
MODEL_CACHE_DIR = "diffusers-cache"


def download_model(model_url: str):
    '''
    Downloads the model from the URL passed in.
    '''
    print(f"Starting model download from: {model_url}")
    
    model_cache_path = Path(MODEL_CACHE_DIR)
    if model_cache_path.exists():
        print(f"Removing existing cache directory: {model_cache_path}")
        shutil.rmtree(model_cache_path)
    model_cache_path.mkdir(parents=True, exist_ok=True)
    print(f"Created cache directory: {model_cache_path}")

    # Check if the URL is from huggingface.co, if so, grab the model repo id.
    parsed_url = urlparse(model_url)
    print(f"Parsed URL - netloc: {parsed_url.netloc}, path: {parsed_url.path}")
    
    if parsed_url.netloc == "huggingface.co":
        model_id = f"{parsed_url.path.strip('/')}"
        print(f"Extracted model ID: {model_id}")
        
        try:
            print("Downloading safety checker...")
            StableDiffusionSafetyChecker.from_pretrained(
                SAFETY_MODEL_ID,
                cache_dir=model_cache_path,
            )
            print("Safety checker downloaded successfully!")

            print(f"Downloading main model: {model_id}")
            StableDiffusionPipeline.from_pretrained(
                model_id,
                cache_dir=model_cache_path,
            )
            print(f"Model {model_id} downloaded successfully!")
            
        except Exception as e:
            print(f"ERROR downloading model: {e}")
            raise e
            
    else:
        print(f"ERROR: Non-HuggingFace URL not supported: {parsed_url.netloc}")
        raise ValueError("Only HuggingFace model URLs are supported")


# ---------------------------------------------------------------------------- #
#                                Parse Arguments                               #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--model_url", type=str,
    default="https://huggingface.co/stabilityai/stable-diffusion-2-1",
    help="URL of the model to download."
)

if __name__ == "__main__":
    args = parser.parse_args()
    download_model(args.model_url)
