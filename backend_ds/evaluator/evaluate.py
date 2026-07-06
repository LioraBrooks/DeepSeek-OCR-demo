# 主评估入口
from .text_align import find_best_match, normalize_text
from .metrics import character_accuracy, sequence_similarity, cer_like
try:
    from ..utils.docx_reader import read_docx
except ImportError:
    from utils.docx_reader import read_docx


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def evaluate(ocr_text_path, gt_text_path, use_alignment=None):
    """Evaluate OCR text against the manually transcribed ground truth."""
    ocr_text = load_text(ocr_text_path)
    gt_text = read_docx(gt_text_path)

    if use_alignment is None:
        # Page-level docx should be compared directly. A full-book docx needs alignment.
        use_alignment = len(normalize_text(gt_text)) > len(normalize_text(ocr_text)) * 2

    if use_alignment:
        matched_gt, align_score = find_best_match(ocr_text, gt_text)
    else:
        matched_gt = gt_text
        align_score = sequence_similarity(ocr_text, gt_text)

    hanzi_accuracy = character_accuracy(ocr_text, matched_gt, mode="hanzi")
    hanzi_similarity = sequence_similarity(ocr_text, matched_gt, mode="hanzi")
    strict_similarity = sequence_similarity(ocr_text, matched_gt, mode="plain")
    strict_error_rate = cer_like(ocr_text, matched_gt, mode="plain")

    evaluation_result = {
        # Main score for ancient-book OCR: ignore punctuation/layout marks.
        "accuracy": round(hanzi_accuracy, 4),
        "similarity": round(hanzi_accuracy, 4),
        "error_rate": round(1 - hanzi_accuracy, 4),

        # Diagnostic scores.
        "hanzi_similarity": round(hanzi_similarity, 4),
        "strict_similarity": round(strict_similarity, 4),
        "punctuation_sensitive_error_rate": round(strict_error_rate, 4),
        "alignment_score": round(align_score, 4),
        "ocr_hanzi_count": len(normalize_text(ocr_text, mode="hanzi")),
        "gt_hanzi_count": len(normalize_text(matched_gt, mode="hanzi")),
        "matched_text": matched_gt[:200],
        "gt_file": gt_text_path,
        "use_alignment": use_alignment
    }

    print(f"OCR路径:{ocr_text_path}")
    print(f"评估结果:{evaluation_result}")

    return evaluation_result
