#自动对齐
import difflib

def sequence_similarity(a, b):
    """基础文本相似度"""
    return difflib.SequenceMatcher(None, a, b).ratio()

def cer_like(a,b):
    """简化版字符错误率(1-相似度)"""
    return 1-sequence_similarity(a,b)