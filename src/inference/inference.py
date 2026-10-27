from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_CONFIGS = {
    "flux2": ("black-forest-labs/FLUX.2-dev", "flux2"),
    "qwen": ("Qwen/Qwen-Image-Edit-2511", "Qwen-Image-edit"),
}


def resolve_model_name(input_name: str) -> tuple[str, str]:
    for name, (model_id, display_name) in MODEL_CONFIGS.items():
        if input_name.lower() in (name.lower(), model_id.lower(), display_name.lower()):
            return model_id, display_name
    return input_name, input_name.replace("/", "_")


def get_inferencer(model: str):
    from src.inference import get_inferencer_class

    model_id, _ = MODEL_CONFIGS[model]
    return get_inferencer_class(model_id)()


def resolve_model_path(model: str, model_path: str | Path | None = None) -> str:
    if model_path is None:
        return MODEL_CONFIGS[model][0]
    directory = Path(model_path).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Local model directory not found: {directory}")
    return str(directory)


def validate_path_component(value, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"{field} must contain only letters, digits, underscores or hyphens")
    return value


def get_output_image_path(output_dir: str | Path, sample: dict) -> Path:
    sample_id = validate_path_component(sample.get("sample_id"), "sample_id")
    task_id = validate_path_component(sample.get("task_id"), "task_id")
    return Path(output_dir).expanduser().resolve() / task_id / sample_id / f"{sample_id}_result.jpg"


def generate_sample(sample: dict, inferencer, image_path: Path, image_root: Path) -> None:
    reference_images = []
    output = None
    try:
        for reference_path in sample["image_paths"]:
            with Image.open(image_root / reference_path) as image:
                reference_images.append(image.convert("RGB"))
        output = inferencer.generate_image(
            prompt=sample["instruction_en"],
            reference_images=reference_images,
            output_width=sample["output_width"],
            output_height=sample["output_height"],
        )
        if output is None:
            raise RuntimeError("Model returned no image")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        with output.convert("RGB") as rgb_output:
            rgb_output.save(image_path, format="JPEG", quality=100)
    finally:
        for image in reference_images:
            image.close()
        if output is not None:
            output.close()


def run_inference(samples: list[dict], model: str, output_dir: str | Path, image_root: Path, weights_source: str) -> int:
    pending = []
    for sample in samples:
        image_path = get_output_image_path(output_dir, sample)
        if not image_path.is_file():
            pending.append((sample, image_path))
    skipped = len(samples) - len(pending)
    print(f"Model: {model}; samples={len(samples)}, skipped={skipped}, pending={len(pending)}")
    if not pending:
        print(f"SUMMARY: success=0, skipped={skipped}, failed=0")
        return 0

    print(f"Loading weights from: {weights_source}")
    inferencer = get_inferencer(model)
    inferencer.load_model(weights_source)
    success = 0
    failed_samples = []
    for index, (sample, image_path) in enumerate(pending, 1):
        print(f"Processing {index}/{len(pending)}: {sample['sample_id']}")
        try:
            generate_sample(sample, inferencer, image_path, image_root)
            success += 1
            print(f"  Saved: {image_path}")
        except Exception as error:
            failed_samples.append(sample["sample_id"])
            print(f"  ERROR: {sample['sample_id']}: {error}", file=sys.stderr)
    print(f"SUMMARY: success={success}, skipped={skipped}, failed={len(failed_samples)}")
    if failed_samples:
        print(f"Failed samples: {', '.join(failed_samples)}", file=sys.stderr)
    return 1 if failed_samples else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate JPG images from a JSONL benchmark", allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=MODEL_CONFIGS, help="One model per run: flux2 or qwen")
    parser.add_argument("--output_dir", required=True, type=Path, help="Save to DIR/TASK/SAMPLE/SAMPLE_result.jpg; no model directory is added")
    parser.add_argument("--manifest", required=True, type=Path, help="JSONL with instruction_en and ordered image_paths")
    parser.add_argument("--image_root", type=Path, help="Root for image_paths (default: manifest parent directory)")
    parser.add_argument("--model_path", type=Path, help="Local Diffusers weights directory (default: model's Hugging Face ID)")
    args = parser.parse_args(argv)

    manifest = args.manifest.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve() if args.image_root is not None else manifest.parent
    try:
        weights_source = resolve_model_path(args.model, args.model_path)
        with manifest.open(encoding="utf-8") as stream:
            samples = [json.loads(line) for line in stream if line.strip()]
    except (OSError, ValueError) as error:
        parser.error(str(error))
    output_dir = args.output_dir.expanduser().resolve()
    print(f"Manifest: {manifest}\nImage root: {image_root}\nOutput directory: {output_dir}")
    try:
        return run_inference(samples, args.model, output_dir, image_root, weights_source)
    except Exception as error:
        print(f"Inference failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
