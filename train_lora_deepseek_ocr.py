#第二版LoRA训练脚本：默认训练参数调保守
import argparse
import json
import os
import random
import time
from pathlib import Path

from deepseek_ocr_training_inputs import build_training_inputs


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def set_seed(seed):
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_dir):
    from transformers import AutoModel, AutoTokenizer

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
    return model, tokenizer


def apply_lora(model, args):
    from peft import LoraConfig, TaskType, get_peft_model

    for param in model.parameters():
        param.requires_grad = False

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(","),
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def move_inputs_to_device(inputs, device):
    import torch

    return {
        "input_ids": inputs["input_ids"].unsqueeze(0).to(device),
        "attention_mask": inputs["attention_mask"].unsqueeze(0).to(device),
        "labels": inputs["labels"].unsqueeze(0).to(device),
        "images": [(inputs["images"][0][0].to(device), inputs["images"][0][1].to(device))],
        "images_seq_mask": inputs["images_seq_mask"].unsqueeze(0).to(device),
        "images_spatial_crop": inputs["images_spatial_crop"].to(device),
        "use_cache": False,
    }


def train(args):
    import torch

    set_seed(args.seed)
    project_root = Path(args.project_root).resolve()
    model_dir = (project_root / args.model_dir).resolve()
    train_path = (project_root / args.train_file).resolve()
    val_path = (project_root / args.val_file).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = read_jsonl(train_path)
    val_records = read_jsonl(val_path) if val_path.exists() else []
    if args.limit_train:
        train_records = train_records[:args.limit_train]
    if args.limit_val:
        val_records = val_records[:args.limit_val]

    print(f"Train records: {len(train_records)}")
    print(f"Val records: {len(val_records)}")

    model, tokenizer = load_model_and_tokenizer(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if device == "cuda":
        model = model.to(torch.bfloat16)

    model = apply_lora(model, args)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    global_step = 0
    losses = []
    started = time.time()

    for epoch in range(args.epochs):
        rng = random.Random(args.seed + epoch)
        rng.shuffle(train_records)
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        for record_index, record in enumerate(train_records, start=1):
            global_step += 1
            image_path = project_root / record["image"]

            inputs = build_training_inputs(
                model=model,
                tokenizer=tokenizer,
                image_path=image_path,
                prompt=record.get("prompt"),
                answer=record["answer"],
                base_size=args.base_size,
                image_size=args.image_size,
                crop_mode=args.crop_mode,
            )
            batch = move_inputs_to_device(inputs, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                output = model(**batch)
                loss = output.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.max_grad_norm
            )
            optimizer.step()

            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)

            if global_step % args.log_every == 0 or record_index == 1:
                recent = losses[-args.log_every:]
                mean_recent = sum(recent) / len(recent)
                print(
                    f"step={global_step} epoch={epoch + 1} item={record_index}/{len(train_records)} "
                    f"loss={loss_value:.4f} mean_recent={mean_recent:.4f}"
                )

            if args.save_every and global_step % args.save_every == 0:
                checkpoint_dir = output_dir / f"checkpoint-step-{global_step}"
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                print(f"Saved checkpoint: {checkpoint_dir}")

        if val_records:
            validate(model, tokenizer, val_records, project_root, args, device, epoch + 1)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    summary = {
        "train_records": len(train_records),
        "val_records": len(val_records),
        "epochs": args.epochs,
        "steps": global_step,
        "mean_train_loss": sum(losses) / len(losses) if losses else None,
        "elapsed_seconds": round(time.time() - started, 1),
        "output_dir": output_dir.as_posix(),
        "target_modules": args.target_modules.split(","),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "learning_rate": args.learning_rate,
    }
    write_json(output_dir / "training_summary.json", summary)
    print("\nTraining finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def validate(model, tokenizer, val_records, project_root, args, device, epoch):
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for record in val_records:
            inputs = build_training_inputs(
                model=model,
                tokenizer=tokenizer,
                image_path=project_root / record["image"],
                prompt=record.get("prompt"),
                answer=record["answer"],
                base_size=args.base_size,
                image_size=args.image_size,
                crop_mode=args.crop_mode,
            )
            batch = move_inputs_to_device(inputs, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                output = model(**batch)
            losses.append(float(output.loss.detach().cpu()))
    mean_loss = sum(losses) / len(losses) if losses else 0.0
    print(f"Validation epoch={epoch} loss={mean_loss:.4f}")
    model.train()


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tune DeepSeek-OCR on paired image/text data.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--model-dir", default="DeepSeek-OCR-model")
    parser.add_argument("--train-file", default="dataset/processed/train.jsonl")
    parser.add_argument("--val-file", default="dataset/processed/val.jsonl")
    parser.add_argument("--output-dir", default="outputs/lora/deepseek_ocr_ancient_books")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--base-size", type=int, default=int(os.getenv("OCR_BASE_SIZE", "1024")))
    parser.add_argument("--image-size", type=int, default=int(os.getenv("OCR_IMAGE_SIZE", "640")))
    parser.add_argument("--crop-mode", action="store_true", default=os.getenv("OCR_CROP_MODE", "false").lower() in ("1", "true", "yes", "on"))
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
