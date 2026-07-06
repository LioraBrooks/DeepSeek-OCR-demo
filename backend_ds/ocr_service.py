import glob
import inspect
import os
import re
import uuid

import torch
from transformers import AutoModel, AutoTokenizer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "DeepSeek-OCR-model"))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

OCR_MAX_NEW_TOKENS = int(os.getenv("OCR_MAX_NEW_TOKENS", "2048"))
OCR_BASE_SIZE = int(os.getenv("OCR_BASE_SIZE", "1024"))
OCR_IMAGE_SIZE = int(os.getenv("OCR_IMAGE_SIZE", "640"))
OCR_CROP_MODE = os.getenv("OCR_CROP_MODE", "false").lower() in ("1", "true", "yes", "on")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DeepSeekOCRService:
    def __init__(self):
        print("Loading DeepSeek-OCR model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            local_files_only=True
        )

        self.model = AutoModel.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            local_files_only=True,
            use_safetensors=True
        )

        self.model = self.model.eval().to(DEVICE)

        if DEVICE == "cuda":
            self.model = self.model.to(torch.bfloat16)

        print("DeepSeek-OCR model loaded.")
        print(
            "OCR config:",
            f"max_new_tokens={OCR_MAX_NEW_TOKENS}",
            f"base_size={OCR_BASE_SIZE}",
            f"image_size={OCR_IMAGE_SIZE}",
            f"crop_mode={OCR_CROP_MODE}"
        )

    def _infer_with_token_limit(self, **kwargs):
        """Clamp generation length for web requests."""
        original_generate = self.model.generate

        def limited_generate(*args, **generate_kwargs):
            current_limit = generate_kwargs.get("max_new_tokens", OCR_MAX_NEW_TOKENS)
            generate_kwargs["max_new_tokens"] = min(current_limit, OCR_MAX_NEW_TOKENS)
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

    def recognize(self, image_path: str):
        prompt = (
            "<image>\n<|grounding|>"
            "Transcribe this ancient Chinese book page as plain text. "
            "If the text is vertical, read columns from right to left and top to bottom. "
            "Only output visible original text. Do not translate or explain. "
            "If a character is unreadable, output □. "
        )
        request_id = uuid.uuid4().hex
        output_path = os.path.join(OUTPUT_DIR, "text", request_id)

        infer_result = self._infer_with_token_limit(
            tokenizer=self.tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=output_path,
            base_size=OCR_BASE_SIZE,
            image_size=OCR_IMAGE_SIZE,
            crop_mode=OCR_CROP_MODE,
            save_results=True,
            test_compress=True
        )

        raw_text = infer_result if isinstance(infer_result, str) and infer_result.strip() else ""

        if not raw_text:
            mmd_files = glob.glob(os.path.join(output_path, "**", "*.mmd"), recursive=True)
            if not mmd_files:
                raise RuntimeError("OCR 未生成 .mmd 结果文件")

            latest_file = max(mmd_files, key=os.path.getctime)
            print("使用mmd文件:", latest_file)

            with open(latest_file, "r", encoding="utf-8") as f:
                raw_text = f.read()

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        history_dir = os.path.join(OUTPUT_DIR, "history")
        os.makedirs(history_dir, exist_ok=True)
        save_path = os.path.join(history_dir, f"{image_name}_{uuid.uuid4().hex[:6]}.mmd")

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(raw_text)

        clean_text = self._clean_text(raw_text)

        save_txt_path = save_path.replace(".mmd", ".txt")
        with open(save_txt_path, "w", encoding="utf-8") as f:
            f.write(clean_text)

        return clean_text

    def _clean_text(self, raw_text):
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

        clean_lines_unique = []
        prev_line = ""
        for line in clean_lines:
            if line != prev_line:
                clean_lines_unique.append(line)
                prev_line = line

        clean_text = "\n".join(clean_lines_unique)
        return self._trim_repetition(clean_text)

    def _trim_repetition(self, text):
        """Cut obvious short-loop degeneration without touching normal repeated prose."""
        compact = re.sub(r"\s+", "", text)
        if len(compact) < 80:
            return text

        for unit_len in range(1, 9):
            tail = compact[-unit_len * 8:]
            if len(tail) == unit_len * 8:
                unit = tail[:unit_len]
                if unit and tail == unit * 8:
                    cutoff = compact.find(unit * 5)
                    if cutoff > 0:
                        return compact[:cutoff + unit_len * 2]
        return text


ocr_service = DeepSeekOCRService()
