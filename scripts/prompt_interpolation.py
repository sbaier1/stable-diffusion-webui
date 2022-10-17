import gradio as gr
import torch

import modules.scripts as scripts
from modules import processing


def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res


class Script(scripts.Script):
    def __init__(self):
        # Override
        self.target_prompt = None
        self.steps = None
        self.interp_func = None

    def title(self):
        return "latent interpolation"

    def ui(self, is_img2img):
        target_prompt = gr.Textbox(label="Target prompt", lines=1)
        steps = gr.Slider(label="Interpolation steps", minimum=1, maximum=150, step=1, value=10)
        interp_func = gr.Textbox(label="Interpolation function", lines=1, value="x")

        return [
            target_prompt,
            steps,
            interp_func,
        ]

    def run(self, p, target_prompt, steps, interp_func):
        # Override
        self.target_prompt = target_prompt
        self.steps = steps
        self.interp_func = interp_func

        orig_sampler_fn = p.sample
        first_latent = None
        target_latent = None
        first = True
        images = []

        def sample_hijack(conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength):
            orig_sampler_res = orig_sampler_fn(conditioning, unconditional_conditioning, seeds, subseeds,
                                               subseed_strength)
            # share the outer variable state
            nonlocal first_latent
            nonlocal target_latent
            nonlocal first
            if first:
                first_latent = orig_sampler_res
                first = False
            else:
                target_latent = orig_sampler_res
            return orig_sampler_res

        # Return arbitrary latents instead of actually sampling at all
        cur_latent = None

        def sample_noop(conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength):
            # share the outer variable state
            nonlocal cur_latent
            return cur_latent

        p.sample = sample_hijack
        # Sample with original prompt first
        print("Sampling the original prompt")
        processed = processing.process_images(p)
        # This is basically the "0th" step of the interpolation
        images.append(processed.images[0])

        p.prompt = self.target_prompt
        # Sample the target prompt now
        print("Sampling the target prompt")
        processed = processing.process_images(p)
        last_image = processed.images

        # Don't do anything when process_images calls sample
        p.sample = sample_noop
        # Interpolate
        print(f"Running latent interpolation for {self.steps} steps")
        for i in range(1, self.steps):
            # Set slerp'd latent for current step
            cur_latent = slerp(i / self.steps, first_latent, target_latent)
            processed = processing.process_images(p)
            images.append(processed.images[0])

        # Last step
        images.append(last_image[0])
        processed.images = images
        return processed
