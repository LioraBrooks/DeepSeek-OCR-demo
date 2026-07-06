# 自动对齐与文本归一化

import re
import unicodedata
from difflib import SequenceMatcher


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def normalize_text(text, mode="plain"):
    """Normalize OCR/GT text before scoring.

    mode="plain": remove whitespace and common layout/markdown noise.
    mode="hanzi": keep only CJK ideographs, ignoring punctuation/layout marks.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("**", "").replace("#", "")

    if mode == "hanzi":
        return "".join(CJK_RE.findall(text))

    text = re.sub(r"\s+", "", text)
    return text


def find_best_match(ocr_text, full_text, step=1):
    """Find the most similar same-length passage in a full reference text."""
    ocr_text = normalize_text(ocr_text)
    full_text = normalize_text(full_text)
    ocr_len = len(ocr_text)

    if not ocr_text or not full_text:
        return "", 0.0

    if len(full_text) <= ocr_len:
        score = SequenceMatcher(None, ocr_text, full_text, autojunk=False).ratio()
        return full_text, score

    best_score = 0.0
    best_text = ""

    max_start = len(full_text) - ocr_len
    for i in range(0, max_start + 1, max(1, step)):
        candidate = full_text[i:i + ocr_len]
        score = SequenceMatcher(None, ocr_text, candidate, autojunk=False).ratio()

        if score > best_score:
            best_score = score
            best_text = candidate

    return best_text, best_score
