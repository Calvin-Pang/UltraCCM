"""Train UltraCCM on in-vivo or ex-vivo micro-ultrasound DICOM data."""

import argparse
import copy
import sys

import torch.distributed as dist

from cm import dist_util, logger
from cm.resample import create_named_schedule_sampler
from cm.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    cm_train_defaults,
    create_ema_and_scales_fn,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from cm.train_util import CMTrainLoop
from scripts.common import load_split, set_seed, validate_case_paths


def create_loaders(args, batch_size):
    train_cases = load_split(args.data_manifest, args.train_split)
    val_cases = load_split(args.data_manifest, args.val_split)
    validate_case_paths(train_cases)
    validate_case_paths(val_cases)

    if args.dataset_mode == "invivo":
        if not isinstance(train_cases, list) or not isinstance(val_cases, list):
            raise TypeError("In-vivo manifest splits must be JSON lists")
        from cm.microus_datasets import load_data

        train_data = load_data(
            root_list=train_cases,
            batch_size=batch_size,
            image_size=args.image_size,
            axis_distance=args.axis_distance,
            scale=args.scale,
            repeat=args.repeat,
            mode="train",
            random_sampling=args.random_sampling,
            adaptive_sampling=args.adaptive_sampling,
            num_workers=args.num_workers,
        )
        val_data = load_data(
            root_list=val_cases,
            batch_size=1,
            image_size=args.image_size,
            axis_distance=args.axis_distance,
            scale=args.scale,
            repeat=1,
            mode="validation",
            random_sampling=args.random_sampling,
            adaptive_sampling=args.adaptive_sampling,
            select_k=args.val_samples,
            num_workers=args.num_workers,
        )
        return train_data, val_data

    if not isinstance(train_cases, dict) or not isinstance(val_cases, dict):
        raise TypeError("Ex-vivo manifest splits must map case paths to [start, stop]")
    if args.paired_supervised and batch_size != 1:
        raise ValueError("Paired ex-vivo training requires a per-rank batch size of 1")

    from cm.microus_datasets_exvivo import load_data

    train_data = load_data(
        root_list=train_cases,
        batch_size=batch_size,
        axis_distance=args.axis_distance,
        scale=args.scale,
        trim=None,
        mode="train",
        self_supervised=not args.paired_supervised,
        num_workers=args.num_workers,
    )
    val_data = load_data(
        root_list=val_cases,
        batch_size=1,
        axis_distance=args.axis_distance,
        scale=args.scale,
        trim=None,
        mode="validation",
        select_k=args.val_samples,
        num_workers=args.num_workers,
    )
    return train_data, val_data


def main():
    args = create_argparser().parse_args()
    if not args.data_manifest:
        raise ValueError("--data_manifest is required")
    if not args.save_dir:
        raise ValueError("--save_dir is required")

    set_seed(args.seed)
    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)
    logger.log("Command used: " + " ".join(sys.argv))

    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=args.target_ema_mode,
        start_ema=args.start_ema,
        scale_mode=args.scale_mode,
        start_scales=args.start_scales,
        end_scales=args.end_scales,
        total_steps=args.total_training_steps,
        distill_steps_per_iter=args.distill_steps_per_iter,
    )
    if args.training_mode == "progdist":
        distillation = False
    elif "consistency" in args.training_mode:
        distillation = True
    else:
        raise ValueError(f"Unknown training mode: {args.training_mode}")

    model_kwargs = args_to_dict(args, model_and_diffusion_defaults().keys())
    model_kwargs["distillation"] = distillation
    model, diffusion = create_model_and_diffusion(**model_kwargs)
    model.to(dist_util.dev()).train()
    if args.use_fp16:
        model.convert_to_fp16()

    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if batch_size < 1 or args.global_batch_size % dist.get_world_size():
            raise ValueError("global_batch_size must be divisible by the MPI world size")
    else:
        batch_size = args.batch_size
    train_data, val_data = create_loaders(args, batch_size)

    teacher_model = None
    teacher_diffusion = None
    if args.teacher_model_path:
        teacher_kwargs = copy.deepcopy(model_kwargs)
        teacher_kwargs["dropout"] = args.teacher_dropout
        teacher_kwargs["distillation"] = False
        teacher_model, teacher_diffusion = create_model_and_diffusion(**teacher_kwargs)
        teacher_model.load_state_dict(
            dist_util.load_state_dict(args.teacher_model_path, map_location="cpu")
        )
        teacher_model.to(dist_util.dev()).eval()
        for destination, source in zip(model.parameters(), teacher_model.parameters()):
            destination.data.copy_(source.data)
        if args.use_fp16:
            teacher_model.convert_to_fp16()

    target_model, _ = create_model_and_diffusion(**model_kwargs)
    target_model.to(dist_util.dev()).train()
    for destination, source in zip(target_model.parameters(), model.parameters()):
        destination.data.copy_(source.data)
    if args.use_fp16:
        target_model.convert_to_fp16()

    CMTrainLoop(
        model=model,
        target_model=target_model,
        teacher_model=teacher_model,
        teacher_diffusion=teacher_diffusion,
        training_mode=args.training_mode,
        ema_scale_fn=ema_scale_fn,
        total_training_steps=args.total_training_steps,
        diffusion=diffusion,
        data=train_data,
        val_data=val_data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=create_named_schedule_sampler(args.schedule_sampler, diffusion),
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        epoch=args.epoch,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_manifest="",
        train_split="train",
        val_split="validation",
        axis_distance=15.0,
        scale=8,
        repeat=10,
        num_workers=8,
        val_samples=5,
        seed=0,
        paired_supervised=False,
        schedule_sampler="uniform",
        lr=5e-6,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2,
        batch_size=-1,
        microbatch=-1,
        ema_rate="0.9999,0.99994,0.9999432189950708",
        log_interval=500,
        save_interval=50000,
        resume_checkpoint="",
        use_fp16=True,
        fp16_scale_growth=1e-3,
        save_dir="",
        epoch=48,
        random_sampling=True,
        adaptive_sampling=False,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cm_train_defaults())
    defaults.update(
        training_mode="consistency_training",
        target_ema_mode="adaptive",
        start_ema=0.95,
        scale_mode="progressive",
        start_scales=2,
        end_scales=150,
        total_training_steps=200000,
        image_size=256,
        num_channels=64,
        num_head_channels=64,
        num_res_blocks=1,
        channel_mult="1,2,4,8,16",
        loss_norm="lpips",
        kl_loss=True,
        use_scale_shift_norm=False,
        resblock_updown=True,
        use_fp16=True,
        weight_schedule="uniform",
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--dataset_mode", choices=("invivo", "exvivo"), default="invivo")
    return parser


if __name__ == "__main__":
    main()
