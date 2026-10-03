"""Lazy model loading and generation; importing the CLI downloads nothing."""

MODEL_IDS = {"sd14": "CompVis/stable-diffusion-v1-4",
             "sdxl": "stabilityai/stable-diffusion-xl-base-1.0"}
REFINER_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"


def load_pipeline(model, model_id, device, refiner_id=None, local_files_only=False):
    import torch
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

    pipeline_class = StableDiffusionPipeline if model == "sd14" else StableDiffusionXLPipeline
    pipe = pipeline_class.from_pretrained(model_id, torch_dtype=torch.float32,
                                          local_files_only=local_files_only).to(device)
    pipe.enable_attention_slicing()
    refiner = None
    if refiner_id:
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            refiner_id, text_encoder_2=pipe.text_encoder_2, vae=pipe.vae,
            torch_dtype=torch.float32, local_files_only=local_files_only).to(device)
        refiner.enable_attention_slicing()
    return pipe, refiner


def encoders(pipe, model):
    pairs = [("text_encoder", pipe.text_encoder, pipe.tokenizer)]
    if model == "sdxl":
        pairs.append(("text_encoder_2", pipe.text_encoder_2, pipe.tokenizer_2))
    return pairs


def generate(pipe, refiner, prompt, seed, steps, guidance, device, split=0.8):
    import torch

    generator = torch.Generator(device=device).manual_seed(seed)
    kwargs = {"prompt": prompt, "num_inference_steps": steps,
              "guidance_scale": guidance, "generator": generator}
    if refiner is None:
        return pipe(**kwargs).images[0]
    latents = pipe(**kwargs, denoising_end=split, output_type="latent").images
    return refiner(**kwargs, denoising_start=split, image=latents).images[0]
