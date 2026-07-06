#探针脚本，读取一条 JSONL 样本，用helper 构造输入并尝试一次 forward(loss)，保证在正式 LoRA 训练前确认多模态训练输入真的能跑通。
#先不训练，只拿 train.jsonl 的第一条样本跑一次 forward()，确认 loss 能不能算出来。
import argparse
import json
import os
from pathlib import Path


def read_first_record(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise RuntimeError(f"No records found in {path}")


def probe(args):
    import torch
    from transformers import AutoModel, AutoTokenizer

    from deepseek_ocr_training_inputs import build_training_inputs

    project_root = Path(args.project_root).resolve()
    dataset_path = (project_root / args.dataset).resolve()
    model_dir = (project_root / args.model_dir).resolve()
    sample = read_first_record(dataset_path)

    print("Sample image:", sample["image"])
    print("Sample ground truth:", sample.get("ground_truth"))
    print("Answer chars:", len(sample["answer"]))

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True
    )
    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)
    if device == "cuda":
        model = model.to(torch.bfloat16)

    inputs = build_training_inputs(
        model=model,
        tokenizer=tokenizer,
        image_path=project_root / sample["image"],
        prompt=sample.get("prompt"),
        answer=sample["answer"],
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=args.crop_mode,
    )

    print("input_ids:", tuple(inputs["input_ids"].shape))
    print("labels:", tuple(inputs["labels"].shape))
    print("supervised tokens:", int((inputs["labels"] != -100).sum().item()))
    print("images_seq_mask:", tuple(inputs["images_seq_mask"].shape), "image tokens:", int(inputs["images_seq_mask"].sum().item()))
    print("images_spatial_crop:", inputs["images_spatial_crop"].tolist())
    print("image crop tensor:", tuple(inputs["images"][0][0].shape))
    print("image ori tensor:", tuple(inputs["images"][0][1].shape))

    batch = {
        "input_ids": inputs["input_ids"].unsqueeze(0).to(device),
        "attention_mask": inputs["attention_mask"].unsqueeze(0).to(device),
        "labels": inputs["labels"].unsqueeze(0).to(device),
        "images": [(inputs["images"][0][0].to(device), inputs["images"][0][1].to(device))],
        "images_seq_mask": inputs["images_seq_mask"].unsqueeze(0).to(device),
        "images_spatial_crop": inputs["images_spatial_crop"].to(device),
        "use_cache": False,
    }

    print("Running forward probe...")
    with torch.no_grad():
        output = model(**batch)

    print("Forward succeeded.")
    print("loss:", output.loss)
    print("logits:", tuple(output.logits.shape))


def parse_args():
    parser = argparse.ArgumentParser(description="Probe one DeepSeek-OCR supervised training step.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset", default="dataset/processed/train.jsonl")
    parser.add_argument("--model-dir", default="DeepSeek-OCR-model")
    parser.add_argument("--base-size", type=int, default=int(os.getenv("OCR_BASE_SIZE", "1024")))
    parser.add_argument("--image-size", type=int, default=int(os.getenv("OCR_IMAGE_SIZE", "640")))
    parser.add_argument("--crop-mode", action="store_true", default=os.getenv("OCR_CROP_MODE", "false").lower() in ("1", "true", "yes", "on"))
    return parser.parse_args()


if __name__ == "__main__":
    probe(parse_args())
