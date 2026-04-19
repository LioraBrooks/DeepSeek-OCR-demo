from pathlib import Path
path = "deepseek_ocr_demo/ground_truth/complete_translation.txt"
with open(path, "rb") as f:
    raw = f.read()

print(raw[:50])