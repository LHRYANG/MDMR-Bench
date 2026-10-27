"""
Evaluate generated images with Gemini against checklist items.

Read annotations from JSONL and locate generated JPGs without inference metadata.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import datetime as dt
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.inference import get_output_image_path

GEMINI_TIMEOUT_MS = 2 * 60 * 1000
DEFAULT_EVAL_MODEL = "gemini-3-pro-preview"
IMAGE_SHORT_SIDE = 1024
EVAL_DIMENSIONS = ("following", "consistency", "composition", "minimality", "quality")

UNIFIED_EVAL_PROMPT = """
Task: Unified Checklist-Based Evaluation for Multi-Reference Image Generation / Editing Benchmark

Objective:
You are an expert evaluator for checklist-based assessment.
Your task is to evaluate whether the output image satisfies each checklist item, and return a structured Y/N decision with concise, evidence-based reasoning.

You must evaluate strictly based on:
1) the instruction and metadata,
2) the Input images,
3) the output image,
4) the provided checklist items.

Do not generate new checklist items. Only evaluate the given checklist.

Scoring Rule:
- "Y" means the checklist item is satisfied.
- "N" means the checklist item is not satisfied.
- Each decision must be independent and evidence-based.
- Do not use uncertain labels. If uncertain, choose the more conservative answer based on visible evidence.

Input JSON fields:
- task_id, instruction_zh, instruction_en, image_paths, checklist
- checklist: a dict with exactly five keys:
  - following: list of checklist items
  - consistency: list of checklist items
  - composition: list of checklist items
  - minimality: list of checklist items
  - quality: list of checklist items

Critical constraints:
1) Evaluate checklist items in their original order within each dimension.
2) Output list lengths must exactly match the corresponding checklist list lengths.
3) Reasoning must be concise, specific, and tied to visible evidence.
4) Do not rely on style preference or subjective taste.
5) Do not output extra keys, summaries, markdown, or explanations outside the required JSON object.

Output format:
Return one valid JSON object with exactly two keys:
{
  "Answer Dict": {
    "following": ["Y" or "N", ...],
    "consistency": ["Y" or "N", ...],
    "composition": ["Y" or "N", ...],
    "minimality": ["Y" or "N", ...],
    "quality": ["Y" or "N", ...]
  },
  "Reason Dict": {
    "following": ["reason for item 1", ...],
    "consistency": ["..."],
    "composition": ["..."],
    "minimality": ["..."],
    "quality": ["..."]
  }
}
Return JSON only. No markdown fence, no extra text.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated benchmark images with Gemini checklist Y/N."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Benchmark JSONL with ordered image_paths and checklist")
    parser.add_argument("--image-root", type=Path, default=None, help="Root for image_paths (default: manifest parent)")
    parser.add_argument(
        "--benchmark-output-root",
        type=Path,
        default=Path("outputs"),
        help="Exact inference --output_dir; no model directory is added",
    )
    parser.add_argument("--max-tries", type=int, default=3)
    parser.add_argument("--sleep-sec", type=float, default=2)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--timeout-ms", type=int, default=GEMINI_TIMEOUT_MS)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Gemini sampling temperature (default: 0.7, matching the Yunwu evaluator).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate annotations and images without API calls or writing files")
    args = parser.parse_args()

    args.manifest = args.manifest.expanduser().resolve()
    args.image_root = args.image_root.expanduser().resolve() if args.image_root else args.manifest.parent
    args.benchmark_output_root = args.benchmark_output_root.expanduser().resolve()
    args.max_tries = max(1, args.max_tries)
    args.sleep_sec = max(0.0, args.sleep_sec)
    args.num_processes = max(1, args.num_processes)
    args.timeout_ms = max(1000, args.timeout_ms)
    return args


def extract_json_block(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    for chunk in fenced:
        try:
            obj = json.loads(chunk.strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    left = text.find("{")
    right = text.rfind("}")
    if left != -1 and right != -1 and right > left:
        obj = json.loads(text[left : right + 1])
        if isinstance(obj, dict):
            return obj

    raise ValueError("Model output is not a valid JSON object.")


def validate_eval_result(eval_obj: dict[str, Any], checklist: dict[str, Any]) -> tuple[bool, str]:
    answer = eval_obj.get("Answer Dict")
    reason = eval_obj.get("Reason Dict")
    if not isinstance(answer, dict) or not isinstance(reason, dict):
        return False, "Missing Answer Dict or Reason Dict."

    for key in EVAL_DIMENSIONS:
        c = checklist.get(key)
        a = answer.get(key)
        r = reason.get(key)
        if not isinstance(c, list):
            return False, f"Checklist key invalid: {key}"
        if not isinstance(a, list) or not isinstance(r, list):
            return False, f"Answer/Reason key invalid: {key}"
        if len(a) != len(r) or len(a) != len(c):
            return False, (
                f"Length mismatch on key {key}: answer={len(a)}, "
                f"reason={len(r)}, checklist={len(c)}"
            )
        for v in a:
            if v not in ("Y", "N"):
                return False, f"Invalid answer value {v!r} under key {key}"
        for v in r:
            if not isinstance(v, str) or not v.strip():
                return False, f"Invalid reason under key {key}"
    return True, ""


def get_augmented_checklist(checklist: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key in EVAL_DIMENSIONS:
        base = checklist.get(key)
        if not isinstance(base, list) or not base:
            raise ValueError(f"Checklist key invalid: {key}")
        if any(not isinstance(question, str) or not question.strip() for question in base):
            raise ValueError(f"Checklist contains an empty or non-string question: {key}")
        out[key] = list(base)
    return out


def is_valid_result_file(path: Path, checklist: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(obj, dict) or obj.get("checklist") != checklist:
        return False
    ok, _ = validate_eval_result(obj, checklist=checklist)
    return ok


def resolve_gemini_endpoint(base_url: str | None, model_name: str) -> tuple[str, str]:
    url = urlsplit(base_url or "https://generativelanguage.googleapis.com")
    if url.scheme not in ("https", "http") or not url.hostname:
        raise ValueError("GEMINI_BASE_URL must be an HTTP(S) URL")
    path = url.path.rstrip("/")
    if "/models/" in path:
        model_name = path.rsplit("/models/", 1)[1].removesuffix(":generateContent")
        if not model_name or "/" in model_name or ":" in model_name:
            raise ValueError("GEMINI_BASE_URL must reference a generateContent model endpoint")
        if not path.endswith(":generateContent"):
            path += ":generateContent"
    else:
        if not path.endswith(("/v1beta", "/v1")):
            path += "/v1beta"
        path += f"/models/{model_name}:generateContent"
    return url._replace(path=path).geturl(), model_name


def _load_image_part(image_path: Path, short_side: int = IMAGE_SHORT_SIDE) -> dict[str, Any]:
    with Image.open(image_path) as source:
        image = source.copy() if source.mode in ("RGB", "RGBA") else source.convert("RGB")
    try:
        width, height = image.size
        if min(width, height) > short_side:
            size = (short_side, int(round(height * short_side / width))) if width <= height else (
                int(round(width * short_side / height)), short_side
            )
            resized = image.resize(size, Image.Resampling.LANCZOS)
            image.close()
            image = resized
        image_format, mime_type = ("PNG", "image/png") if image_path.suffix.lower() == ".png" else ("JPEG", "image/jpeg")
        if image_format == "JPEG" and image.mode == "RGBA":
            converted = image.convert("RGB")
            image.close()
            image = converted
        with io.BytesIO() as buffer:
            if image_format == "JPEG":
                image.save(buffer, format=image_format, quality=95)
            else:
                image.save(buffer, format=image_format)
            data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"inlineData": {"mimeType": mime_type, "data": data}}
    finally:
        image.close()


def build_eval_parts(
    checklist_obj: dict[str, Any],
    checklist_for_eval: dict[str, list[str]],
    image_root: Path,
    generated_image_path: Path,
) -> list[dict[str, Any]]:
    payload = {
        "sample_id": checklist_obj.get("sample_id"),
        "task_id": checklist_obj.get("task_id"),
        "subtask_id": checklist_obj.get("subtask_id"),
        "instruction_zh": checklist_obj.get("instruction_zh"),
        "instruction_en": checklist_obj.get("instruction_en"),
        "image_paths": checklist_obj["image_paths"],
        "output_width": checklist_obj["output_width"],
        "output_height": checklist_obj["output_height"],
        "checklist": checklist_for_eval,
        "output_image": str(generated_image_path),
    }

    parts = [
        {"text": UNIFIED_EVAL_PROMPT},
        {"text": "Evaluate the following sample JSON and attached images."},
        {"text": json.dumps(payload, ensure_ascii=False, indent=2)},
        {"text": "Now I will provide Input images in image_paths list order (Image 1, Image 2, ...)."},
    ]
    for index, image_path in enumerate(checklist_obj["image_paths"], 1):
        parts.append({"text": f"Input image {index}:"})
        parts.append(_load_image_part(image_root / image_path))
    parts.append({"text": "Output image (to evaluate):"})
    parts.append(_load_image_part(generated_image_path))
    parts.append({"text": "Return only the JSON object with keys 'Answer Dict' and 'Reason Dict'."})
    return parts


def call_gemini_eval(
    api_key: str,
    base_url: str,
    timeout_ms: int,
    parts: list[dict[str, Any]],
    checklist: dict[str, list[str]],
    sample_id: str,
    temperature: float,
    max_tries: int,
    sleep_sec: float,
    raw_json_path: Path,
) -> tuple[dict[str, Any], str]:
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": temperature, "topP": 1, "maxOutputTokens": 8192},
    }
    headers = {"Content-Type": "application/json"}
    key = api_key.strip().removeprefix("Bearer ").strip()
    if urlsplit(base_url).hostname == "generativelanguage.googleapis.com":
        headers["x-goog-api-key"] = key
    else:
        headers["Authorization"] = f"Bearer {key}"
    required_counts = {dimension: len(checklist[dimension]) for dimension in EVAL_DIMENSIONS}
    raw_records = json.loads(raw_json_path.read_text(encoding="utf-8")) if raw_json_path.exists() else []
    if not isinstance(raw_records, list):
        raise ValueError(f"Expected a response list in {raw_json_path}")
    raw_json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_raw = raw_json_path.with_suffix(".json.tmp")

    last_error: Exception | None = None
    for attempt in range(1, max_tries + 1):
        raw_record = {
            "attempt": attempt,
            "time": dt.datetime.now(dt.timezone.utc).isoformat(),
            "response": None,
        }
        try:
            with requests.post(base_url, headers=headers, json=payload, timeout=timeout_ms / 1000) as response:
                raw_record["status_code"] = response.status_code
                raw_record["response"] = response.text
                try:
                    raw_record["response"] = response.json()
                except ValueError:
                    pass
                if not response.ok:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
                data = raw_record["response"]
                if not isinstance(data, dict):
                    raise ValueError("Gemini response is not a JSON object")
            candidate = data["candidates"][0]
            if candidate.get("finishReason") == "MAX_TOKENS":
                raise ValueError("Gemini response truncated: finishReason=MAX_TOKENS")
            response_parts = candidate["content"]["parts"]
            raw_text = "\n".join(
                part["text"].strip() for part in response_parts
                if not part.get("thought") and part.get("text")
            ).strip()
            parsed = extract_json_block(raw_text)
            ok, message = validate_eval_result(parsed, checklist)
            if not ok:
                payload["contents"][0]["parts"] = parts + [{
                    "text": f"Return a complete corrected evaluation. The previous response failed validation: {message}. "
                    "Include Answer Dict and Reason Dict with all five dimensions. "
                    f"Required list lengths in each dict: {json.dumps(required_counts)}. "
                    'Use only the strings "Y" or "N" for answers and one non-empty reason per answer.'
                }]
                raise ValueError(f"Invalid evaluation structure: {message}")
            return parsed, raw_text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            raw_record["error"] = str(exc)
        finally:
            raw_records.append(raw_record)
            temporary_raw.write_text(json.dumps(raw_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary_raw.replace(raw_json_path)
        if attempt < max_tries:
            print(f"{sample_id}: attempt {attempt}/{max_tries} failed; retrying | {last_error}", flush=True)
            time.sleep(sleep_sec * attempt)
    raise RuntimeError(f"Gemini evaluation failed after retries: {last_error}")


def build_result_json(
    checklist_obj: dict[str, Any],
    checklist_for_eval: dict[str, list[str]],
    eval_obj: dict[str, Any],
    raw_text: str,
    job: dict[str, Any],
    generated_image_path: Path,
) -> dict[str, Any]:
    result = dict(checklist_obj)
    result["checklist"] = checklist_for_eval
    result["source_manifest"] = str(job["manifest"])
    result["image_root"] = str(job["image_root"])
    result["generated_image_path"] = str(generated_image_path)
    result["Answer Dict"] = eval_obj.get("Answer Dict")
    result["Reason Dict"] = eval_obj.get("Reason Dict")
    result["evaluation_model"] = str(job["eval_model_name"])
    result["evaluation_generated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    result["evaluation_raw_response"] = raw_text
    return result


def process_one(job: dict[str, Any]) -> dict[str, Any]:
    result_json_path = Path(job["result_json_path"])
    generated_image_path = Path(job["generated_image_path"])
    index = int(job["index"])

    record: dict[str, Any] = {
        "index": index,
        "sample_id": job.get("sample_id"),
        "manifest": str(job["manifest"]),
        "generated_image": str(generated_image_path),
        "result_json": str(result_json_path),
        "status": "pending",
    }

    temporary_result = result_json_path.with_suffix(".json.tmp")
    try:
        checklist_obj = job["sample"]
        checklist_for_eval = get_augmented_checklist(checklist_obj["checklist"])

        if not generated_image_path.is_file():
            record["status"] = "missing_output"
            record["reason"] = f"Generated image file not found: {generated_image_path}"
            return record

        if not job["dry_run"] and not job["overwrite"] and is_valid_result_file(
            result_json_path, checklist=checklist_for_eval
        ):
            record["status"] = "skipped_done"
            return record

        parts = build_eval_parts(
            checklist_obj=checklist_obj,
            checklist_for_eval=checklist_for_eval,
            image_root=Path(job["image_root"]),
            generated_image_path=generated_image_path,
        )
        if job["dry_run"]:
            record["status"] = "validated"
            return record
        eval_obj, raw_text = call_gemini_eval(
            api_key=str(job["api_key"]),
            base_url=job.get("base_url"),
            timeout_ms=int(job["timeout_ms"]),
            parts=parts,
            checklist=checklist_for_eval,
            sample_id=str(job["sample_id"]),
            temperature=float(job["temperature"]),
            max_tries=int(job["max_tries"]),
            sleep_sec=float(job["sleep_sec"]),
            raw_json_path=result_json_path.parent / "raw.json",
        )

        result_obj = build_result_json(
            checklist_obj=checklist_obj,
            checklist_for_eval=checklist_for_eval,
            eval_obj=eval_obj,
            raw_text=raw_text,
            job=job,
            generated_image_path=generated_image_path,
        )
        result_json_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_result.write_text(
            json.dumps(result_obj, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_result.replace(result_json_path)
        record["status"] = "ok"
        return record
    except Exception as exc:  # noqa: BLE001
        record["status"] = "failed"
        record["error"] = str(exc)
        return record
    finally:
        if not job["dry_run"]:
            temporary_result.unlink(missing_ok=True)


def run_jobs_parallel(jobs: list[dict[str, Any]], num_processes: int) -> list[dict[str, Any]]:
    if len(jobs) == 0:
        return []
    results: list[dict[str, Any]] = []
    counts = {status: 0 for status in ("ok", "failed", "missing_output", "skipped_done", "validated")}

    def report(result: dict[str, Any]) -> None:
        results.append(result)
        counts[result["status"]] += 1
        message = (
            f"[{len(results)}/{len(jobs)}] {result['sample_id']}: {result['status']} | "
            f"success={counts['ok']} failed={counts['failed']} "
            f"missing={counts['missing_output']} skipped={counts['skipped_done']}"
        )
        if counts["validated"]:
            message += f" validated={counts['validated']}"
        if result["status"] in ("failed", "missing_output"):
            message += f" | {result.get('error') or result.get('reason')}"
        print(message, flush=True)

    if num_processes == 1:
        for job in jobs:
            report(process_one(job))
        return results
    with cf.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {executor.submit(process_one, job): job for job in jobs}
        for future in cf.as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                job = futures[future]
                result = {"index": job["index"], "sample_id": job["sample_id"], "status": "failed", "error": str(error)}
            report(result)
    return results


def main() -> int:
    args = parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    base_url, eval_model_name = resolve_gemini_endpoint(
        os.getenv("GEMINI_BASE_URL"), os.getenv("GEMINI_MODEL", DEFAULT_EVAL_MODEL)
    )

    with args.manifest.open(encoding="utf-8") as stream:
        samples = [json.loads(line) for line in stream if line.strip()]
    print(f"Loaded {len(samples)} manifest records; images: {args.benchmark_output_root}")
    print(f"Evaluation model: {eval_model_name}")
    jobs: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        sample_id = sample["sample_id"]
        generated_image_path = get_output_image_path(args.benchmark_output_root, sample)
        result_json_path = generated_image_path.parent / f"{sample_id}_results.json"
        jobs.append(
            {
                "index": idx,
                "sample_id": sample_id,
                "sample": sample,
                "manifest": str(args.manifest),
                "image_root": str(args.image_root),
                "generated_image_path": str(generated_image_path),
                "result_json_path": str(result_json_path),
                "eval_model_name": eval_model_name,
                "max_tries": args.max_tries,
                "sleep_sec": args.sleep_sec,
                "timeout_ms": args.timeout_ms,
                "temperature": args.temperature,
                "overwrite": args.overwrite,
                "dry_run": args.dry_run,
                "api_key": api_key or "",
                "base_url": base_url,
            }
        )

    results = run_jobs_parallel(jobs=jobs, num_processes=args.num_processes)
    results.sort(key=lambda x: int(x.get("index", 0)))

    ok_n = sum(1 for r in results if r.get("status") == "ok")
    skip_n = sum(1 for r in results if r.get("status") == "skipped_done")
    miss_n = sum(1 for r in results if r.get("status") == "missing_output")
    fail_n = sum(1 for r in results if r.get("status") == "failed")

    if args.dry_run:
        validated_n = sum(1 for result in results if result.get("status") == "validated")
        print(f"dry-run: validated={validated_n} missing_output={miss_n} failed={fail_n}; no files written")
        return 1 if miss_n or fail_n else 0

    print(
        f"done. total={len(results)} ok={ok_n} skipped_done={skip_n} "
        f"missing_output={miss_n} failed={fail_n}"
    )
    return 1 if fail_n or miss_n else 0


if __name__ == "__main__":
    raise SystemExit(main())
