# RunPod IMG2IMG Serverless Endpoint

Custom RunPod serverless worker that supports both txt2img and img2img transformations using Stable Diffusion 2.1.

## Features
- **IMG2IMG Support**: Transform input images (faces) into fantasy creatures
- **TXT2IMG Support**: Generate images from text prompts only
- **Dual Mode**: Automatically detects input type and routes accordingly
- **Optimized**: Uses xformers for memory efficiency

## Deployment
Set `MODEL_URL=https://huggingface.co/stabilityai/stable-diffusion-2-1` environment variable.
