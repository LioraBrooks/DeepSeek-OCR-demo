import difflib

from .text_align import normalize_text


def sequence_similarity(a, b, mode="plain"):
    """SequenceMatcher similarity after text normalization."""
    a = normalize_text(a, mode=mode)
    b = normalize_text(b, mode=mode)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def edit_distance(a, b):
    """Levenshtein edit distance for character-level OCR scoring."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost
            ))
        previous = current
    return previous[-1]


def character_accuracy(ocr_text, gt_text, mode="hanzi"):
    """Character accuracy: 1 - edit_distance / reference_length."""
    ocr_text = normalize_text(ocr_text, mode=mode)
    gt_text = normalize_text(gt_text, mode=mode)
    if not ocr_text and not gt_text:
        return 1.0
    if not gt_text:
        return 0.0

    distance = edit_distance(ocr_text, gt_text)
    return max(0.0, 1 - distance / len(gt_text))


def cer_like(a, b, mode="plain"):
    """Simplified error rate."""
    return 1 - sequence_similarity(a, b, mode=mode)
