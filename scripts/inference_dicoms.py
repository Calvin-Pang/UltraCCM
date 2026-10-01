"""Run UltraCCM inference on in-vivo or ex-vivo micro-ultrasound DICOM cases."""

import argparse
import time
from pathlib import Path

import numpy as np
import pydicom
import torch
from PIL import Image
from tqdm import tqdm

from cm import dist_util, logger
from cm.karras_diffusion import karras_sample
from cm.microus_datasets import MicroUSAxialImageFolder, MicroUSExvivoImageFolder
from cm.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from cm.utils import polar2cartesian_fast
from scripts.common import load_split, set_seed, validate_case_paths


def prepare_output(root, name, save_dicoms):
    image_dir = root / name / "imgs"
    image_dir.mkdir(parents=True, exist_ok=True)
    dicom_dir = root / name / "dicoms"
    if save_dicoms:
        dicom_dir.mkdir(parents=True, exist_ok=True)
    return image_dir, dicom_dir


def create_dicom(template, image, label, series_uid, study_uid, instance_number, spacing):
    dataset = template.copy()
    dataset.remove_private_tags()
    for keyword in (
        "PatientBirthDate",
        "PatientSex",
        "AccessionNumber",
        "InstitutionName",
        "ReferringPhysicianName",
        "StudyID",
    ):
        if hasattr(dataset, keyword):
            setattr(dataset, keyword, "")
    dataset.PatientName = label
    dataset.PatientID = label
    dataset.SeriesDescription = label
    dataset.SOPInstanceUID = pydicom.uid.generate_uid()
    dataset.SeriesInstanceUID = series_uid
    dataset.StudyInstanceUID = study_uid
    if hasattr(dataset, "file_meta"):
        dataset.file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
    dataset.SliceLocation = float(instance_number) * spacing
    dataset.SliceThickness = spacing
    dataset.ImagePositionPatient = [0, 0, float(instance_number) * spacing]
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.InstanceNumber = instance_number
    dataset.Rows, dataset.Columns = image.shape
    dataset.PixelData = image.tobytes()
    return dataset


def save_result(
    image,
    image_dir,
    dicom_dir,
    template,
    case_id,
    kind,
    slice_number,
    save_dicoms,
    series_uid,
    study_uid,
    spacing,
):
    filename = f"case_{case_id}_slice_{slice_number:04d}_{kind}"
    Image.fromarray(image).convert("L").save(image_dir / f"{filename}.png")
    if save_dicoms:
        dataset = create_dicom(
            template,
            image,
            f"anonymous_{kind}",
            series_uid,
            study_uid,
            slice_number,
            spacing,
        )
        dataset.save_as(dicom_dir / f"{filename}.dcm")


def build_dataset(args, path, bounds):
    if args.dataset_mode == "invivo":
        return MicroUSAxialImageFolder(
            root_path=path,
            axis_distance=args.axis_distance,
            scale=args.scale,
        )
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError(f"Ex-vivo case '{path}' needs [start, stop] trim bounds")
    return MicroUSExvivoImageFolder(
        root_path=path,
        axis_distance=args.axis_distance,
        scale=args.scale,
        trim=bounds,
        unit_theta=args.unit_theta,
    )


def main():
    args = create_argparser().parse_args()
    if not args.data_manifest:
        raise ValueError("--data_manifest is required")
    if not args.model_path:
        raise ValueError("--model_path is required")

    cases = load_split(args.data_manifest, args.test_split)
    validate_case_paths(cases)
    if args.dataset_mode == "invivo" and not isinstance(cases, list):
        raise TypeError("The in-vivo test split must be a JSON list")
    if args.dataset_mode == "exvivo" and not isinstance(cases, dict):
        raise TypeError("The ex-vivo test split must map paths to [start, stop]")

    set_seed(args.seed)
    dist_util.setup_dist()
    output_root = Path(args.save_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    logger.configure(dir=str(output_root))

    model_kwargs = args_to_dict(args, model_and_diffusion_defaults().keys())
    model_kwargs["distillation"] = "consistency" in args.training_mode
    model, diffusion = create_model_and_diffusion(**model_kwargs)
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev()).eval()
    if args.use_fp16:
        model.convert_to_fp16()

    case_items = [(path, None) for path in cases] if isinstance(cases, list) else cases.items()
    total_time = 0.0
    total_slices = 0
    with torch.no_grad():
        for case_path, bounds in case_items:
            case_id = Path(case_path).name
            logger.log(f"Processing case {case_id}")
            dataset = build_dataset(args, case_path, bounds)
            indices = list(range(0, len(dataset), args.slice_stride))
            if args.max_slices > 0:
                indices = indices[: args.max_slices]

            case_root = output_root / case_id
            ref_dirs = prepare_output(case_root, "ref", args.save_dicoms)
            sr_dirs = prepare_output(case_root, "sr", args.save_dicoms)
            hr_dirs = (
                prepare_output(case_root, "hr", args.save_dicoms)
                if args.dataset_mode == "exvivo"
                else None
            )
            template = pydicom.dcmread(dataset.filenames[0]) if args.save_dicoms else None
            uids = {
                kind: (pydicom.uid.generate_uid(), pydicom.uid.generate_uid())
                for kind in ("ref", "sr", "hr")
            }
            spacing = (
                float(getattr(template, "PixelSpacing", [1.0])[0])
                if template and args.dataset_mode == "invivo"
                else 1.0
            )

            for index in tqdm(indices, desc=case_id):
                item = dataset[index]
                lr_grids = item["lr_grids"].unsqueeze(0).to(dist_util.dev())
                hr_grids = item["hr_grids"].unsqueeze(0).to(dist_util.dev())
                hr_inte = item["hr_inte"].unsqueeze(0).to(dist_util.dev())
                height, width = hr_inte.shape[-2:]

                started = time.perf_counter()
                sample = karras_sample(
                    diffusion,
                    model,
                    (1, 1, height, width),
                    steps=args.steps,
                    hr_inte=hr_inte,
                    lr_grids=lr_grids,
                    meta_info=item["meta_info"],
                    model_kwargs={},
                    device=dist_util.dev(),
                    clip_denoised=True,
                    sampler=args.sampler,
                    generator=None,
                    ts=tuple(int(value) for value in args.ts.split(",")) if args.ts else None,
                )
                total_time += time.perf_counter() - started
                total_slices += 1

                ref = polar2cartesian_fast(
                    ((hr_inte + 1) / 2).clamp(0, 1),
                    hr_grids,
                    item["meta_info"]["h_expand"],
                )[0]
                sr = polar2cartesian_fast(
                    ((sample + 1) / 2).clamp(0, 1),
                    hr_grids,
                    item["meta_info"]["h_expand"],
                )[0]
                if args.dataset_mode == "invivo":
                    ref = np.flip(ref, axis=-1)
                    sr = np.flip(sr, axis=-1)

                slice_number = index + 1
                save_result(ref, *ref_dirs, template, case_id, "ref", slice_number,
                            args.save_dicoms, *uids["ref"], spacing)
                save_result(sr, *sr_dirs, template, case_id, "sr", slice_number,
                            args.save_dicoms, *uids["sr"], spacing)

                if args.dataset_mode == "exvivo":
                    hr_tensor = ((item["hr_img"].unsqueeze(0).to(dist_util.dev()) + 1) / 2).clamp(0, 1)
                    hr = polar2cartesian_fast(
                        hr_tensor, hr_grids, item["meta_info"]["h_expand"]
                    )[0]
                    save_result(hr, *hr_dirs, template, case_id, "hr", slice_number,
                                args.save_dicoms, *uids["hr"], spacing)

    if total_slices:
        logger.log(f"Average model time: {total_time / total_slices:.5f} seconds per slice")


def create_argparser():
    defaults = model_and_diffusion_defaults()
    defaults.update(
        training_mode="consistency_distillation",
        data_manifest="",
        test_split="test",
        model_path="",
        save_dir="outputs/inference",
        seed=42,
        sampler="onestep",
        steps=151,
        ts="",
        axis_distance=15.0,
        scale=8,
        unit_theta=1.0,
        slice_stride=4,
        max_slices=0,
        save_dicoms=False,
        image_size=256,
        num_channels=64,
        num_head_channels=64,
        num_res_blocks=1,
        channel_mult="1,2,4,8,16",
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
