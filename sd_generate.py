import argparse
import json
import numpy as np
import os
import time
import torch

from diffusers import StableDiffusionPipeline
from diffusers.utils import logging
from diffusers.models.attention_processor import AttnProcessor2_0

from SD_setup.token_reduc_sd import apply_patch
from SD_setup.class_names import IM_CLASS_DICT

def get_args_parser():
    parser = argparse.ArgumentParser('Stable Diffusion evaluation script for ToMe and ATC', add_help=False)

    parser.add_argument('--img_per_class', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--output_dir_base', type=str, default="./SD_generations")
    parser.add_argument('--use_flash_attn', action="store_true")
    parser.add_argument('--use_full_precision', action="store_true")
    
    # Shared Arguments
    parser.add_argument('--merge_fn_str', type=str, default="tome", choices=["tome", "atc", "none"])
    parser.add_argument('--reduction_ratio', type=float, default=0.5)
    parser.add_argument('--max_downsample', type=int, default=1)
    parser.add_argument('--merge_attn', action="store_true")
    parser.add_argument('--merge_crossattn', action="store_true")
    parser.add_argument('--merge_mlp', action="store_true")
    
    # ToMe Arguments
    parser.add_argument('--sx', type=int, default=2)
    parser.add_argument('--sy', type=int, default=2)
    parser.add_argument('--use_rand', action="store_true")

    # ATC Arguments
    parser.add_argument('--linkage', default="average", type=str)

    return parser

def main(args):
    total_start = time.time()

    logging.disable_progress_bar()

    ## Setup Pipeline
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.use_full_precision: 
        precision = torch.float32
    else:
        precision = torch.float16
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker = None, requires_safety_checker=False, torch_dtype=precision).to(device)
    
    if args.use_flash_attn:
        pipe.unet.set_attn_processor(AttnProcessor2_0())
    else:
        pipe.unet.set_default_attn_processor()

    args.scheduler = type(pipe.scheduler).__name__

    pipe.set_progress_bar_config(**{"disable": True})

    # Apply merging function with provided args
    # Can also use pipe.unet in place of pipe here
    if args.merge_fn_str != "none":
        apply_patch(pipe, 
                    ratio=args.reduction_ratio,
                    max_downsample=args.max_downsample,
                    sx=args.sx,
                    sy=args.sy,
                    use_rand=args.use_rand,
                    merge_attn=args.merge_attn,
                    merge_crossattn=args.merge_crossattn,
                    merge_mlp=args.merge_mlp,
                    merge_fn_str=args.merge_fn_str,
                    atc_linkage=args.linkage
                    ) 
    
    # Setup directory for images
    if args.merge_fn_str == "none":
        dir_prefix = "stablediffusion"
    else:
        dir_prefix = args.merge_fn_str + "_" + str(args.reduction_ratio) + "_" + str(args.max_downsample) + "_" + str(args.merge_attn) + "_" + str(args.merge_crossattn) + "_" + str(args.merge_mlp)

    dir_prefix = time.strftime("%d%b%Y_%H%M%S") + "_" + dir_prefix

    if args.merge_fn_str == "tome":
        dir_suffix = "_" + str(args.sx) + "_" + str(args.sy) + "_" + str(args.use_rand)
    elif args.merge_fn_str == "atc":
        dir_suffix = "_" + args.linkage
    else:
        dir_suffix = ""

    if args.use_flash_attn:
        dir_suffix += "_Flash"

    output_dir = os.path.join(args.output_dir_base, dir_prefix + dir_suffix)
    os.makedirs(output_dir, exist_ok=True)

    args.output_dir = output_dir
    args.durations = []

    print(dir_prefix + dir_suffix)
    prompt_base = "A high quality photograph of a "
    for cls_idx, cls in IM_CLASS_DICT.items():
        prompt = prompt_base + cls + "."
        for img_idx in range(args.img_per_class):
            generator = torch.Generator(device).manual_seed(args.seed+int(str(cls_idx)+str(img_idx)))

            start = time.time()
            output = pipe(prompt, num_inference_steps=args.num_inference_steps, guidance_scale = args.guidance_scale, generator=generator)
            end = time.time()

            image = output.images[0]
            image.save(os.path.join(output_dir, f"{cls_idx}_{img_idx}.png"))
            args.durations.append(end-start) 
        
    args.duration_mean = np.mean(args.durations)
    args.duration_std = np.std(args.durations)

    with open(os.path.join(args.output_dir_base, dir_prefix + dir_suffix + ".json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    total_end = time.time()

    print(total_end-total_start)
    print()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Stable Diffusion evaluation script for ToMe and ATC', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
