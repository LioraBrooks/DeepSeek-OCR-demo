#自动对齐

from difflib import SequenceMatcher

def find_best_match(ocr_text, full_text, step=50):
    """在全文中找到最相似片段"""
    ocr_len = len(ocr_text)

    best_score = 0
    best_text = ""

    for i in range(0, len(full_text) - ocr_len, step):
        candidate = full_text[i:i + ocr_len]
        score = SequenceMatcher(None, ocr_text, candidate).ratio()

        if score > best_score:
            best_score = score
            best_text = candidate

    return best_text,best_score