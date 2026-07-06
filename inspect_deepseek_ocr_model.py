#检查 DeepSeek-OCR 模型结构，找出哪些模块适合挂 LoRA
import argparse
from collections import Counter
from pathlib import Path


def module_summary(model):
    rows = []
    type_counter = Counter()

    for name, module in model.named_modules():
        if not name:
            continue
        module_type = type(module).__name__
        type_counter[module_type] += 1

        if any(key in name.lower() for key in (
            "attn", "attention", "mlp", "linear", "proj", "vision", "encoder", "decoder", "language"
        )):
            param_count = sum(p.numel() for p in module.parameters(recurse=False))
            rows.append((name, module_type, param_count))

    return rows, type_counter


def print_lora_candidates(rows):
    keywords = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "linear", "proj")
    candidates = []

    for name, module_type, param_count in rows:
        lower = name.lower()
        if module_type == "Linear" or any(key in lower for key in keywords):
            candidates.append((name, module_type, param_count))

    print("\nLikely LoRA candidate modules:")
    for name, module_type, param_count in candidates[:300]:
        print(f"{name}\t{module_type}\tparams={param_count}")

    if len(candidates) > 300:
        print(f"... {len(candidates) - 300} more candidates omitted")


def inspect_model(args):
    from transformers import AutoModel, AutoTokenizer

    model_dir = Path(args.model_dir).resolve()
    print(f"Loading tokenizer from {model_dir}")
    AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True
    )

    print(f"Loading model from {model_dir}")
    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True
    )
    model = model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel class:", type(model).__name__)
    print("Total parameters:", total_params)
    print("Trainable parameters before freezing:", trainable_params)

    rows, type_counter = module_summary(model)

    print("\nTop module types:")
    for module_type, count in type_counter.most_common(40):
        print(f"{module_type}: {count}")

    print("\nRelevant modules:")
    for name, module_type, param_count in rows[:500]:
        print(f"{name}\t{module_type}\tparams={param_count}")
    if len(rows) > 500:
        print(f"... {len(rows) - 500} more relevant modules omitted")

    print_lora_candidates(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect DeepSeek-OCR model modules for LoRA planning.")
    parser.add_argument("--model-dir", default="DeepSeek-OCR-model")
    return parser.parse_args()


if __name__ == "__main__":
    inspect_model(parse_args())
