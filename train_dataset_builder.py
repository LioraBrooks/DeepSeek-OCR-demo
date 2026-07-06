import argparse
import json
import random
from pathlib import Path

from docx import Document


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."


def read_docx_text(path):
    doc = Document(path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()


def collect_files(directory, extensions):
    files = {}
    if not directory.exists():
        return files

    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            files[path.stem] = path
    return files


def make_record(image_path, docx_path, project_root, prompt):
    answer = read_docx_text(docx_path)
    if not answer:
        return None

    return {
        "image": image_path.relative_to(project_root).as_posix(),
        "ground_truth": docx_path.relative_to(project_root).as_posix(),
        "prompt": prompt,
        "answer": answer,
    }


def split_records(records, train_ratio, val_ratio, seed):
    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)

    total = len(records)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    train = records[:train_count]
    val = records[train_count:train_count + val_count]
    test = records[train_count + val_count:]
    return train, val, test


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_dataset(args):
    project_root = Path(args.project_root).resolve()
    images_dir = (project_root / args.images_dir).resolve()
    gt_dir = (project_root / args.ground_truth_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    images = collect_files(images_dir, IMAGE_EXTENSIONS)
    docx_files = collect_files(gt_dir, {".docx"})

    paired_names = sorted(set(images) & set(docx_files))
    missing_docx = sorted(set(images) - set(docx_files))
    missing_images = sorted(set(docx_files) - set(images))

    records = []
    skipped_empty = []
    for name in paired_names:
        record = make_record(images[name], docx_files[name], project_root, args.prompt)
        if record is None:
            skipped_empty.append(name)
            continue
        records.append(record)

    train, val, test = split_records(records, args.train_ratio, args.val_ratio, args.seed)

    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "val.jsonl", val)
    write_jsonl(output_dir / "test.jsonl", test)

    summary = {
        "images_dir": images_dir.as_posix(),
        "ground_truth_dir": gt_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "image_count": len(images),
        "docx_count": len(docx_files),
        "paired_count": len(paired_names),
        "usable_count": len(records),
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "missing_docx_count": len(missing_docx),
        "missing_image_count": len(missing_images),
        "skipped_empty_count": len(skipped_empty),
        "missing_docx": missing_docx,
        "missing_images": missing_images,
        "skipped_empty": skipped_empty,
    }
    write_json(output_dir / "summary.json", summary)

    print("Dataset build finished.")
    print(f"Images: {len(images)}")
    print(f"DOCX: {len(docx_files)}")
    print(f"Paired: {len(paired_names)}")
    print(f"Usable: {len(records)}")
    print(f"Train/Val/Test: {len(train)}/{len(val)}/{len(test)}")
    print(f"Summary: {output_dir / 'summary.json'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build DeepSeek-OCR fine-tuning JSONL dataset.")
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument("--images-dir", default="dataset/images", help="Image directory under project root.")
    parser.add_argument("--ground-truth-dir", default="dataset/ground_truth", help="DOCX directory under project root.")
    parser.add_argument("--output-dir", default="dataset/processed", help="Output directory under project root.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt saved into each JSONL record.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
