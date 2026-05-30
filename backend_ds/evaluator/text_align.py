#自动对齐

import re
from difflib import SequenceMatcher


def normalize_text(text):
    """去掉空白字符，避免换行和段落格式影响 OCR 评分。"""
    return re.sub(r"\s+", "", text or "")


def find_best_match(ocr_text, full_text, step=1):
    """在全文中找到与 OCR 文本最相似的片段。"""
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
