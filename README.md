# Text-to-Image-Conversion-Using-Stable-Diffusion.
Deep Learning and Edge AI
https://colab.research.google.com/drive/19EZlQ3Xq9eDI4BtXu-7mEO-qINBU5rSE#scrollTo=wsHnNvQEDXqw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class TextToImageModel:
    def __init__(self, device='cuda', model_id="runwayml/stable-diffusion-v1-5"):
        """
        Initialize the Text to Image Conversion model using Stable Diffusion
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            model_id: Pre-trained stable diffusion model ID
        """
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        print(f"Using device: {self.device}")
        
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(self.device)
        
       
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32)
        self.pipe = self.pipe.to(self.device)
    
    def encode_text(self, prompt):
        """
        Encode text prompt to latent representation using CLIP model
        
        Args:
            prompt: Text prompt to encode
            
        Returns:
            Text embedding tensor
        """
       
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
  
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input_ids)[0]
        
        return text_embeddings
    
    def generate_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5):
        """
        Generate initial random latents and prepare for the diffusion process
        
        Args:
            text_embeddings: Encoded text embeddings
            height: Output image height
            width: Output image width
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance
            
        Returns:
            Initial latents for the diffusion process
        """
     
        latents_shape = (1, self.pipe.unet.in_channels, height // 8, width // 8)
        latents = torch.randn(latents_shape, device=self.device, dtype=torch.float16 if self.device == 'cuda' else torch.float32)
        
      
        latents = latents * self.pipe.scheduler.init_noise_sigma
        
        return latents
    
    def denoise_latents(self, latents, text_embeddings, num_inference_steps=50, guidance_scale=7.5):
        """
        Run the denoising diffusion process to generate the image
        
        Args:
            latents: Initial latent vectors
            text_embeddings: Encoded text embeddings
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance
            
        Returns:
            Denoised latents representing the generated image
        """
       
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
   
        uncond_input = self.tokenizer([""], padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt")
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
       
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
       
        for i, t in tqdm(enumerate(self.pipe.scheduler.timesteps)):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            
            
            with torch.no_grad():
                noise_pred = self.pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
           
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
        
        return latents
    
    def decode_latents(self, latents):
        """
        Decode the latent representation to an image using VAE
        
        Args:
            latents: Latent representation to decode
            
        Returns:
            Decoded image as a numpy array
        """
        # Scale and decode the latents using VAE
        latents = 1 / 0.18215 * latents
        
        with torch.no_grad():
            image = self.pipe.vae.decode(latents).sample
        
        # Convert to numpy array
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        
        # Convert to uint8 format
        image = (image * 255).round().astype("uint8")
        
        return image[0]  # Return the first image
    
    def generate_image(self, prompt, height=512, width=512, num_inference_steps=50, guidance_scale=7.5):
        """
        Generate an image from a text prompt using the full pipeline
        
        Args:
            prompt: Text prompt to generate image from
            height: Output image height
            width: Output image width
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance
            
        Returns:
            Generated image as a numpy array
        """
        # Encode text to embedding
        text_embeddings = self.encode_text(prompt)
        
        # Generate initial latents
        latents = self.generate_latents(
            text_embeddings,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        # Denoise latents
        denoised_latents = self.denoise_latents(
            latents,
            text_embeddings,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        # Decode latents to image
        image = self.decode_latents(denoised_latents)
        
        return image
    
    def generate_image_pipeline(self, prompt, height=512, width=512, num_inference_steps=50, guidance_scale=7.5):
        """
        Generate an image using the built-in pipeline (faster but less customizable)
        
        Args:
            prompt: Text prompt to generate image from
            height: Output image height
            width: Output image width
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance
            
        Returns:
            Generated image
        """
        with torch.no_grad():
            image = self.pipe(
                prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            
        return image


# Example usage
def main():
    # Initialize the model
    model = TextToImageModel(device='cuda')
    
    # Generate an image using the detailed process
    prompt = "A sunset over the ocean with beautiful orange and red colors"
    image = model.generate_image(prompt, num_inference_steps=30)
    
    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Generated Image: "{prompt}"')
    plt.show()
    
    # Alternatively, use the pipeline (faster)
    # image = model.generate_image_pipeline(prompt, num_inference_steps=30)
    # image.save("generated_image.png")


if __name__ == "__main__":
    main()
