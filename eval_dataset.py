#评估脚本,既能跑 baseline，也能跑 LoRA
import argparse
import glob
import inspect
import json
import os
import re
import time
import uuid
from pathlib import Path

from backend_ds.evaluator.metrics import character_accuracy, cer_like, sequence_similarity
from backend_ds.evaluator.text_align import normalize_text


DEFAULT_PROMPT = (
    "<image>\n<|grounding|>"
    "Transcribe this ancient Chinese book page as plain text. "
    "If the text is vertical, read columns from right to left and top to bottom. "
    "Only output visible original text. Do not translate or explain. "
    "If a character is unreadable, output □. "
)


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_ocr_text(raw_text):
    clean_lines = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith(("system", "user", "assistant")):
            continue
        if line.startswith("![]") or line.startswith("<"):
            continue
        line = line.replace("**", "").replace("#", "")
        line = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", line)
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"[|`*_~\[\](){}]", "", line)
        line = line.strip()
        if line:
            clean_lines.append(line)

    unique_lines = []
    prev_line = ""
    for line in clean_lines:
        if line != prev_line:
            unique_lines.append(line)
            prev_line = line
    return "\n".join(unique_lines)


class DeepSeekOCREvaluator:
    def __init__(
        self,
        model_dir,
        output_dir,
        max_new_tokens,
        base_size,
        image_size,
        crop_mode,
        adapter_dir=None,
        merge_adapter=False,
    ):
        self.model_dir = Path(model_dir).resolve()
        self.adapter_dir = Path(adapter_dir).resolve() if adapter_dir else None
        self.output_dir = Path(output_dir).resolve()
        self.max_new_tokens = max_new_tokens
        self.base_size = base_size
        self.image_size = image_size
        self.crop_mode = crop_mode
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading DeepSeek-OCR model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True,
            local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True,
            local_files_only=True,
            use_safetensors=True
        )
        if self.adapter_dir:
            from peft import PeftModel

            print(f"Loading LoRA adapter from {self.adapter_dir}...")
            self.model = PeftModel.from_pretrained(
                self.model,
                str(self.adapter_dir),
                local_files_only=True
            )
            if merge_adapter:
                print("Merging LoRA adapter into base model for evaluation...")
                self.model = self.model.merge_and_unload()
        self.model = self.model.eval().to(self.device)
        if self.device == "cuda":
            self.model = self.model.to(torch.bfloat16)
        print(f"Model loaded on {self.device}.")

    def _infer_with_token_limit(self, **kwargs):
        original_generate = self.model.generate

        def limited_generate(*args, **generate_kwargs):
            current_limit = generate_kwargs.get("max_new_tokens", self.max_new_tokens)
            generate_kwargs["max_new_tokens"] = min(current_limit, self.max_new_tokens)
            if generate_kwargs.get("do_sample") is False:
                generate_kwargs.pop("temperature", None)
            return original_generate(*args, **generate_kwargs)

        self.model.generate = limited_generate
        try:
            infer_kwargs = dict(kwargs)
            infer_params = inspect.signature(self.model.infer).parameters
            if "eval_mode" in infer_params:
                infer_kwargs["eval_mode"] = True
            if "verbose" in infer_params:
                infer_kwargs["verbose"] = False
            return self.model.infer(**infer_kwargs)
        finally:
            self.model.generate = original_generate

    def recognize(self, image_path, prompt):
        request_id = uuid.uuid4().hex
        output_path = self.output_dir / request_id
        output_path.mkdir(parents=True, exist_ok=True)

        infer_result = self._infer_with_token_limit(
            tokenizer=self.tokenizer,
            prompt=prompt,
            image_file=str(image_path),
            output_path=str(output_path),
            base_size=self.base_size,
            image_size=self.image_size,
            crop_mode=self.crop_mode,
            save_results=True,
            test_compress=True
        )

        raw_text = infer_result if isinstance(infer_result, str) and infer_result.strip() else ""
        if not raw_text:
            mmd_files = glob.glob(str(output_path / "**" / "*.mmd"), recursive=True)
            if not mmd_files:
                raise RuntimeError(f"OCR did not generate .mmd output for {image_path}")
            latest_file = max(mmd_files, key=os.path.getctime)
            with open(latest_file, "r", encoding="utf-8") as f:
                raw_text = f.read()

        return clean_ocr_text(raw_text)


def score_prediction(prediction, answer):
    accuracy = character_accuracy(prediction, answer, mode="hanzi")
    return {
        "accuracy": round(accuracy, 4),
        "error_rate": round(1 - accuracy, 4),
        "hanzi_similarity": round(sequence_similarity(prediction, answer, mode="hanzi"), 4),
        "strict_similarity": round(sequence_similarity(prediction, answer, mode="plain"), 4),
        "punctuation_sensitive_error_rate": round(cer_like(prediction, answer, mode="plain"), 4),
        "ocr_hanzi_count": len(normalize_text(prediction, mode="hanzi")),
        "gt_hanzi_count": len(normalize_text(answer, mode="hanzi")),
    }


def summarize(records, elapsed_seconds):
    scored = [r for r in records if r.get("status") == "ok"]
    failed = [r for r in records if r.get("status") != "ok"]

    def mean(key):
        if not scored:
            return 0.0
        return round(sum(r["metrics"][key] for r in scored) / len(scored), 4)

    return {
        "total": len(records),
        "ok": len(scored),
        "failed": len(failed),
        "elapsed_seconds": round(elapsed_seconds, 1),
        "mean_accuracy": mean("accuracy"),
        "mean_error_rate": mean("error_rate"),
        "mean_hanzi_similarity": mean("hanzi_similarity"),
        "mean_strict_similarity": mean("strict_similarity"),
    }


def evaluate_dataset(args):
    project_root = Path(args.project_root).resolve()
    dataset_path = (project_root / args.dataset).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    model_dir = (project_root / args.model_dir).resolve()
    adapter_dir = (project_root / args.adapter_dir).resolve() if args.adapter_dir else None
    result_prefix = args.result_prefix or ("lora" if adapter_dir else "baseline")

    records = list(read_jsonl(dataset_path))
    if args.limit:
        records = records[:args.limit]

    evaluator = DeepSeekOCREvaluator(
        model_dir=model_dir,
        output_dir=output_dir / "raw_outputs",
        max_new_tokens=args.max_new_tokens,
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=args.crop_mode,
        adapter_dir=adapter_dir,
        merge_adapter=args.merge_adapter,
    )

    results = []
    started = time.time()

    for index, sample in enumerate(records, start=1):
        image_path = (project_root / sample["image"]).resolve()
        prompt = args.prompt or sample.get("prompt") or DEFAULT_PROMPT
        answer = sample["answer"]
        print(f"[{index}/{len(records)}] {sample['image']}")

        item_started = time.time()
        try:
            prediction = evaluator.recognize(image_path, prompt)
            metrics = score_prediction(prediction, answer)
            result = {
                "status": "ok",
                "image": sample["image"],
                "ground_truth": sample.get("ground_truth"),
                "prediction": prediction,
                "answer": answer,
                "metrics": metrics,
                "elapsed_seconds": round(time.time() - item_started, 1),
            }
            print(f"  accuracy={metrics['accuracy']} strict={metrics['strict_similarity']}")
        except Exception as exc:
            result = {
                "status": "error",
                "image": sample["image"],
                "ground_truth": sample.get("ground_truth"),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": round(time.time() - item_started, 1),
            }
            print(f"  ERROR {type(exc).__name__}: {exc}")
        results.append(result)

        write_jsonl(output_dir / f"{result_prefix}_results.jsonl", results)

    summary = summarize(results, time.time() - started)
    summary["model_dir"] = model_dir.as_posix()
    summary["adapter_dir"] = adapter_dir.as_posix() if adapter_dir else None
    summary["dataset"] = dataset_path.as_posix()
    write_json(output_dir / f"{result_prefix}_summary.json", summary)
    print("Evaluation finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DeepSeek-OCR on a JSONL dataset.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset", default="dataset/processed/test.jsonl")
    parser.add_argument("--output-dir", default="dataset/eval/baseline")
    parser.add_argument("--model-dir", default="DeepSeek-OCR-model")
    parser.add_argument("--adapter-dir", default=None, help="Optional LoRA adapter directory.")
    parser.add_argument("--merge-adapter", action="store_true", help="Merge LoRA adapter before inference.")
    parser.add_argument("--result-prefix", default=None, help="Output filename prefix; defaults to baseline or lora.")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("OCR_MAX_NEW_TOKENS", "2048")))
    parser.add_argument("--base-size", type=int, default=int(os.getenv("OCR_BASE_SIZE", "1024")))
    parser.add_argument("--image-size", type=int, default=int(os.getenv("OCR_IMAGE_SIZE", "640")))
    parser.add_argument("--crop-mode", action="store_true", default=os.getenv("OCR_CROP_MODE", "false").lower() in ("1", "true", "yes", "on"))
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_dataset(parse_args())
