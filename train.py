import json
import copy
import gc
import hashlib
import itertools
import logging
import math
import os
import shutil
import warnings
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from transformers import CLIPTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import open_clip
import torch.nn as nn


import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
# from diffusers.loaders import (
#     LoraLoaderMixin,
#     text_encoder_lora_state_dict,
# )
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
# from diffusers.models.lora import LoRALinearLayer
from diffusers.optimization import get_scheduler
# from diffusers.training_utils import unet_lora_state_dict
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from ID_MULTI import ID_MULTI
from dataset import SubjectDrivenTextToImageDataset, collate_fn
from transformers.activations import QuickGELUActivation as QuickGELU
from utils import train_parse_args
from safetensors.torch import save_file

logger = get_logger(__name__)

def train(args):

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        device_placement=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.logging_dir is not None:
            os.makedirs(args.logging_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )
    
    # Make one log on every process with the configuration for debugging.
    t = time.localtime()
    str_m_d_y_h_m_s = time.strftime("%m-%d-%Y_%H-%M-%S", t)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(args.logging_dir, f"{str_m_d_y_h_m_s}.log")
            ),
        ]
        if accelerator.is_main_process
        else [],
    )

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        print("Seed: ", args.seed)
        set_seed(args.seed)

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    
    # Load model
    model = ID_MULTI(args)
    # with open('/LAVIS/multi_BlipDisenBooth/multi_BLIP_DisenBooth.txt', 'w') as f:
    #         for name, param in model.named_parameters():
    #             f.write(f"{name}:\n")
    #             f.write(f"{param}\n")
    # return
    # if args.load_model is not None:
    #     model.load_state_dict(
    #         torch.load(Path(args.load_model) / "pytorch_model.bin", map_location="cpu")
    #     )
    if args.train_text_encoder:
        model.text_encoder.requires_grad(args.train_text_encoder)

    model.to(accelerator.device, dtype = weight_dtype)
    # model.to(accelerator.device)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            logging.info("Using xformers.")
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    optimizer_class = torch.optim.AdamW

    unet_params = list([p for p in model.unet.parameters() if p.requires_grad])
    
    if args.train_text_encoder:
        text_encoder_params = list([p for p in model.text_encoder.parameters() if p.requires_grad])
    else:
        optimizer_params = [
            {"params": unet_params, "lr": args.learning_rate * args.unet_lr_scale},
        ]

    if args.train_text_encoder:
        text_encoder_params = [p for p in model.text_encoder.parameters() if p.requires_grad]
        optimizer_params.append({"params": text_encoder_params, "lr": args.learning_rate * args.text_encoder_lr_scale})

    optimizer = optimizer_class(
        optimizer_params,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    logging.info("Building datasets...")
    # tokenizer = CLIPTokenizer.from_pretrained(
    #         args.pretrained_model_name_or_path, subfolder="tokenizer"
    #     )

    print(args.subject_text)   # 獲取當前保留的內存量
    train_dataset = SubjectDrivenTextToImageDataset(
        image_dir=args.instance_data_dir,
        subject_text=args.subject_text,
        text_prompt=args.text_prompt,
        questions=args.questions,
    )


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args.train_batch_size,
        shuffle = True,
        collate_fn = lambda samples: collate_fn(samples),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        tracker_config.pop("validation_images")
        tracker_config.pop("questions")
        subject_text = ""
        if isinstance(tracker_config['subject_text'], list):
            for subject in tracker_config['subject_text']:
                subject_text = ",".join([subject_text, subject])
            tracker_config['subject_text'] = subject_text
        accelerator.init_trackers(
            project_name="BLIP-DisenBooth",
            config=tracker_config
        )
    
    # Calculate th total and training parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)



    # Train!
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         logger.info(f"Trainable parameter: {name} with shape {param.shape}")

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("\n***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total parameters number = {total_params}")
    logger.info(f"  Training parameters number = {trainable_params}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save

    initial_global_step = 0
    
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            initial_global_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

            # move all the state to the correct device
            model.to(accelerator.device)
            if args.use_ema:
                model.module.ema_param.to(accelerator.device)

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    model.before_training(model=model, dataset=train_dataset)
    
    
    from tqdm.contrib.logging import logging_redirect_tqdm
    # print(list(train_dataloader))
    # return
    with logging_redirect_tqdm():
        for epoch in range(first_epoch, args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                progress_bar.set_description("Global step: {}".format(global_step))

                with accelerator.accumulate(model), torch.backends.cuda.sdp_kernel(
                    enable_flash=not args.disable_flashattention
                ):
                    # if step == 0:
                    #     model.before_training(model=model,
                    #                           dataset=batch
                    #                           )
                    return_dict = model(batch)
                    loss = return_dict['loss']

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = (
                            unet_params +text_encoder_params
                            if args.train_text_encoder
                            else unet_params
                        )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    # if accelerator.is_main_process:
                    #     if global_step % args.checkpointing_steps == 0:
                    #         # _before_ saving state, check if this save would set us over the `checkpoints_total_limit
                    #         save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    #         accelerator.save_state(save_path)
                    #         logger.info(f"Saved state to {save_path}")
                
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                # logging.info(f"{logs}")

                if global_step >= args.max_train_steps:
                    break
            
            if accelerator.is_main_process:
                if args.validation is True and global_step % args.validation_epochs == 0:
                    # save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}", "images")
                    text_prompt = [
                    'in the jungle',
                    'in the snow',
                    'on the beach',
                    'on a cobblestone street',
                    'on top of pink fabric',
                    'on top of a wooden floor',
                    'with a city in the background',
                    'with a mountain in the background',
                    'with a blue house in the background',
                    'on top of a purple rug in a forest',
                    'with a wheat field in the background',
                    'with a tree and autumn leaves in the background',
                    'with the Eiffel Tower in the background',
                    'floating on top of water',
                    'floating in an ocean of milk',
                    'on top of green grass with sunflowers around it',
                    'on top of a mirror',
                    'on top of the sidewalk in a crowded street',
                    'on top of a dirt road',
                    'on top of a white rug',
                    'red',
                    'purple',
                    'shiny',
                    'wet',
                    'cube shaped']
                    num_output = 4
                    iter_seed = 8888
                    guidance_scale = 7.5
                    num_inference_steps = 100
                    negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"
                    
                    for prompt in text_prompt:
                        logger.info(
                        f"Running validation... \n Generating {num_output} images with prompt:"
                        f"{prompt}."
                        )
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}"+"/images/"+prompt.replace(' ', '_')+"/" )
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)

                        samples = {
                            "ref_images": None,
                            "cond_images": None,
                            "cond_subject": [args.subject_text],
                            "tgt_subject": [args.subject_text],
                            "prompt": [prompt],
                            "subjects_position": [list(train_dataloader)[0]['subjects_position'][0]]
                        }
                        if prompt in {'red','purple','shiny','wet','cube shaped'}:
                            samples["tgt_subject"] = [f"{prompt} {args.subject_text[0]}"]
                            samples["prompt"] = [""]
                            data = {str(i): f"a {prompt} {args.subject_text[0]}" for i in range(4)}
                        else:
                            data = {str(i): f"a {args.subject_text[0]} {prompt}" for i in range(4)}

                        json_file_path = os.path.join(save_path, 'dataset.json')
                        with open(json_file_path, 'w') as json_file:
                            json.dump(data, json_file, indent=4)
                        
                        for i in range(num_output):
                            # TODO: 處理samples[subject]為list狀況
                            output = model.generate(
                                samples,
                                seed=iter_seed + i,
                                guidance_scale=guidance_scale,
                                num_inference_steps=num_inference_steps,
                                neg_prompt=negative_prompt,
                                height=512,
                                width=512,
                            )

                            img = output[0]
                            
                            img.save(os.path.join(save_path, f"{i}.png"))
                            logging.info(f"Image have been saved in {save_path}")
                    torch.cuda.empty_cache()
                
    # Save the every layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)

        torch.save(model.state_dict(), f"{args.output_dir}/model_weights.pth")

    accelerator.end_training()

    # 初始化模型的某些參數的梯度為零以模擬沒有更新的情況
    for param in model.parameters():
        if torch.rand(1).item() > 0.5:
            param.grad = torch.zeros_like(param)

    # 計算實際有更新的參數量
    updated_params = sum(p.numel() for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"實際訓練更新過的參數量: {updated_params}")

if __name__ == "__main__":
    args = train_parse_args()
    train(args)