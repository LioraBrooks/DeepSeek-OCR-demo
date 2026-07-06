#检查模型的 forward() 签名和 tokenizer 输出格式
import argparse
import inspect
import json
from pathlib import Path


def safe_signature(obj):
    try:
        return str(inspect.signature(obj))
    except Exception as exc:
        return f"<signature unavailable: {type(exc).__name__}: {exc}>"


def safe_source_file(obj):
    try:
        return inspect.getsourcefile(obj) or "<unknown>"
    except Exception as exc:
        return f"<source file unavailable: {type(exc).__name__}: {exc}>"


def safe_source_preview(obj, max_lines=80):
    try:
        source = inspect.getsource(obj)
    except Exception as exc:
        return [f"<source unavailable: {type(exc).__name__}: {exc}>"]
    return source.splitlines()[:max_lines]


def list_interesting_methods(model):
    keywords = (
        "forward",
        "infer",
        "encode",
        "image",
        "vision",
        "prepare",
        "embed",
        "loss",
        "generate",
        "chat",
        "token",
    )
    names = []
    for name in dir(model):
        lower = name.lower()
        if any(key in lower for key in keywords):
            attr = getattr(model, name, None)
            if callable(attr):
                names.append(name)
    return sorted(set(names))


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def inspect_model(args):
    from transformers import AutoModel, AutoTokenizer

    model_dir = Path(args.model_dir).resolve()
    print(f"Model dir: {model_dir}")

    print_section("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True
    )
    print("Tokenizer class:", type(tokenizer).__name__)
    print("Tokenizer source:", safe_source_file(type(tokenizer)))
    print("pad_token:", tokenizer.pad_token, "pad_token_id:", tokenizer.pad_token_id)
    print("eos_token:", tokenizer.eos_token, "eos_token_id:", tokenizer.eos_token_id)
    print("bos_token:", tokenizer.bos_token, "bos_token_id:", tokenizer.bos_token_id)

    print_section("Loading Model")
    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True
    )
    model.eval()
    print("Model class:", type(model).__name__)
    print("Model source:", safe_source_file(type(model)))
    print("Config class:", type(model.config).__name__)

    config_dict = model.config.to_dict() if hasattr(model.config, "to_dict") else {}
    interesting_config = {
        key: value
        for key, value in config_dict.items()
        if any(token in key.lower() for token in ("model", "vision", "image", "token", "hidden", "layer", "vocab"))
    }
    print("Interesting config keys:")
    print(json.dumps(interesting_config, ensure_ascii=False, indent=2, default=str)[:6000])

    print_section("Core Signatures")
    print("model.forward:", safe_signature(model.forward))
    if hasattr(model, "infer"):
        print("model.infer:", safe_signature(model.infer))
    if hasattr(model, "generate"):
        print("model.generate:", safe_signature(model.generate))

    print_section("Forward Source Preview")
    for line in safe_source_preview(model.forward, max_lines=args.source_lines):
        print(line)

    if hasattr(model, "infer"):
        print_section("Infer Source Preview")
        for line in safe_source_preview(model.infer, max_lines=args.source_lines):
            print(line)

    print_section("Interesting Callable Methods")
    for name in list_interesting_methods(model):
        attr = getattr(model, name)
        print(f"{name}: {safe_signature(attr)}")

    print_section("Named Child Modules")
    for name, module in model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        print(f"{name}\t{type(module).__name__}\tparams={param_count}")

    if args.try_text_forward:
        print_section("Optional Text-only Forward Probe")
        prompt = args.prompt
        answer = args.answer
        text = prompt + answer
        encoded = tokenizer(text, return_tensors="pt")
        print("Tokenizer output keys:", list(encoded.keys()))
        print("input_ids shape:", tuple(encoded["input_ids"].shape))

        labels = encoded["input_ids"].clone()
        try:
            output = model(**encoded, labels=labels)
            print("Forward succeeded with labels.")
            print("Output type:", type(output).__name__)
            if hasattr(output, "loss"):
                print("loss:", output.loss)
            if hasattr(output, "logits"):
                print("logits shape:", tuple(output.logits.shape))
        except Exception as exc:
            print("Forward with labels failed.")
            print("Error type:", type(exc).__name__)
            print("Error:", exc)
            print("This is expected if the model requires image embeddings or custom multimodal inputs.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect DeepSeek-OCR forward/infer signatures before LoRA training."
    )
    parser.add_argument("--model-dir", default="DeepSeek-OCR-model")
    parser.add_argument("--source-lines", type=int, default=80)
    parser.add_argument("--try-text-forward", action="store_true")
    parser.add_argument(
        "--prompt",
        default="<image>\n<|grounding|>Convert the document to markdown.\n"
    )
    parser.add_argument("--answer", default="測試文字")
    return parser.parse_args()


if __name__ == "__main__":
    inspect_model(parse_args())
