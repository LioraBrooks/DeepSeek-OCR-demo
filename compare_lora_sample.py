#单样本诊断
import argparse
import gc
import json
from pathlib import Path

from eval_dataset import DeepSeekOCREvaluator, read_jsonl, score_prediction


def mean_metric(items, section, key):
    values = [item[section]["metrics"][key] for item in items]
    return round(sum(values) / len(values), 4) if values else 0.0


def compare(args):
    project_root = Path(args.project_root).resolve()
    dataset_path = (project_root / args.dataset).resolve()
    records = list(read_jsonl(dataset_path))
    if not records:
        raise RuntimeError(f"No records found in {dataset_path}")

    if args.limit:
        selected = records[args.index:args.index + args.limit]
    else:
        selected = [records[args.index]]

    baseline = DeepSeekOCREvaluator(
        model_dir=project_root / args.model_dir,
        output_dir=project_root / args.output_dir / "baseline_raw",
        max_new_tokens=args.max_new_tokens,
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=args.crop_mode,
    )

    baseline_outputs = []
    for offset, sample in enumerate(selected, start=args.index):
        image_path = (project_root / sample["image"]).resolve()
        prompt = args.prompt or sample.get("prompt")
        answer = sample["answer"]
        print(f"Baseline sample {offset}: {sample['image']}")
        text = baseline.recognize(image_path, prompt)
        baseline_outputs.append({
            "prediction": text,
            "metrics": score_prediction(text, answer),
        })

    del baseline
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    lora = DeepSeekOCREvaluator(
        model_dir=project_root / args.model_dir,
        output_dir=project_root / args.output_dir / "lora_raw",
        max_new_tokens=args.max_new_tokens,
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=args.crop_mode,
        adapter_dir=project_root / args.adapter_dir,
        merge_adapter=args.merge_adapter,
    )

    results = []
    for item_index, sample in enumerate(selected):
        image_path = (project_root / sample["image"]).resolve()
        prompt = args.prompt or sample.get("prompt")
        answer = sample["answer"]
        actual_index = args.index + item_index
        print(f"LoRA sample {actual_index}: {sample['image']}")
        lora_text = lora.recognize(image_path, prompt)
        lora_output = {
            "prediction": lora_text,
            "metrics": score_prediction(lora_text, answer),
        }
        result = {
            "index": actual_index,
            "image": sample["image"],
            "ground_truth": sample.get("ground_truth"),
            "baseline": baseline_outputs[item_index],
            "lora": lora_output,
        }
        results.append(result)
        print("  Baseline metrics:", baseline_outputs[item_index]["metrics"])
        print("  LoRA metrics:", lora_output["metrics"])

    summary = {
        "total": len(results),
        "baseline_mean_accuracy": mean_metric(results, "baseline", "accuracy"),
        "lora_mean_accuracy": mean_metric(results, "lora", "accuracy"),
        "baseline_mean_hanzi_similarity": mean_metric(results, "baseline", "hanzi_similarity"),
        "lora_mean_hanzi_similarity": mean_metric(results, "lora", "hanzi_similarity"),
        "baseline_mean_strict_similarity": mean_metric(results, "baseline", "strict_similarity"),
        "lora_mean_strict_similarity": mean_metric(results, "lora", "strict_similarity"),
    }

    output_path = project_root / args.output_dir / "compare_sample.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print("Summary:", summary)
    print("Wrote:", output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare baseline and LoRA OCR output on one sample.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset", default="dataset/processed/test.jsonl")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Compare multiple samples starting at --index.")
    parser.add_argument("--model-dir", default="DeepSeek-OCR-model")
    parser.add_argument("--adapter-dir", default="outputs/lora/full_epoch1")
    parser.add_argument("--output-dir", default="dataset/eval/lora_diagnostics")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--crop-mode", action="store_true")
    parser.add_argument("--merge-adapter", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    compare(parse_args())
