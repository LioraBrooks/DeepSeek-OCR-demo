#主评估入口
import os
from .text_align import find_best_match, normalize_text
from .metrics import sequence_similarity, cer_like
from utils.docx_reader import read_docx 

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def evaluate(ocr_text_path, gt_text_path, use_alignment=None):
    """OCR vs 译本评估"""
    ocr_text = load_text(ocr_text_path)
    gt_text = read_docx(gt_text_path)

    if use_alignment is None:
        # 分页 docx 通常长度接近 OCR 结果，直接比较更可靠；
        # full.docx 这类全文才需要自动寻找片段。
        use_alignment = len(normalize_text(gt_text)) > len(normalize_text(ocr_text)) * 2

    if use_alignment:
        matched_gt, align_score = find_best_match(ocr_text, gt_text)
    else:
        matched_gt = gt_text
        align_score = sequence_similarity(ocr_text, gt_text)

    #全局相似度
    global_score = sequence_similarity(ocr_text, matched_gt)

    #错误率
    error_rate = cer_like(ocr_text, matched_gt)

    #构造要返回的字典
    evaluation_result={
        "alignment_score": round(align_score, 4),           #对齐评分
        "similarity": round(global_score, 4),               #全局相似度
        "error_rate": round(error_rate, 4),                  #错误率
        "matched_text": matched_gt[:200],                    #匹配的文本片段
        "gt_file": gt_text_path,
        "use_alignment": use_alignment
    }
    
    #打印调试
    print(f"OCR路径:{ocr_text_path}")
    print(f"评估结果:{evaluation_result}")
    
    return evaluation_result
