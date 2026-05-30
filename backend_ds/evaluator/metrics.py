#自动对齐
import difflib

from .text_align import normalize_text


def sequence_similarity(a, b):
    """基础文本相似度"""
    a = normalize_text(a)
    b = normalize_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def cer_like(a, b):
    """简化版字符错误率 1-相似度"""
    return 1 - sequence_similarity(a, b)
