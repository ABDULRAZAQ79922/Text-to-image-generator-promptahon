import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import gradio as gr

def generate_image(prompt, output_path="output_image.png", num_inference_steps=50, guidance_scale=7.5):
    """
    Creating an image from a text prompt using Stable Diffusion.

    Parameters:
        prompt (str): The text describing the image you want to create.
        output_path (str, optional): Where to save the generated image file. Default is "output_image.png".
        num_inference_steps (int, optional): How many steps to take during the image generation process. More steps can improve the quality but take longer.
        guidance_scale (float, optional): Adjusts how closely the output matches the prompt. Higher values mean closer adherence.

    Returns:
        Image: The generated image as a PIL Image object.
    """
    # Using the model "CompVis/stable-diffusion-v1-4"
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loading the Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    # Generating the image
    with torch.autocast(device):
        generated_image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]

    # image saving
    generated_image.save(output_path)

    return generated_image

def generate_image_interface(prompt):
    """
    Wrapper for Gradio to generate an image based on user input.
    
    Parameters:
        prompt (str): The text prompt for the image generation.
        
    Returns:
        Image: The generated image.
    """
    return generate_image(prompt)

best_prompt = (
    "A city landscape experiencing the effects of an urban heat island. The image should show a densely populated urban area "
    "with heatwaves radiating from asphalt roads and concrete buildings. The city appears hazy due to smog and pollution, "
    "with people visibly struggling with heat-related issues like heatstroke. In the background, cooler green spaces can be seen "
    "on the city's outskirts, symbolizing lower temperatures away from the urban core. The color scheme should use intense reds "
    "and oranges for the hot city, transitioning to cooler greens and blues outside."
)

if __name__ == "__main__":
    # using Gradio interface
    gr.Interface(
        fn=generate_image_interface,
        inputs="text",
        outputs="image",
        title="Urban Heat Island Impact Generator",
        description=(
            "Create an AI-generated image illustrating the effects of urban heat islands on public health. "
            "It highlights the temperature difference between the heat-affected urban area and the cooler surrounding regions."
        ),
        examples=[[best_prompt]]
    ).launch()
