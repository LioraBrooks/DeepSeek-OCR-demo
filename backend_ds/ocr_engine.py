# from transformers import AutoModel, AutoTokenizer
# import torch

# _model=None
# _tokenizer=None

# def get_engine():
#     global _model
#     global _tokenizer
#     if _model is None:
#         model_name="deepseek-ai/DeepSeek-OCR"

#         _tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
#         _model = AutoModel.from_pretrained(
#             model_name,
#             trust_remote_code=True
#         ).eval()

#     return _model, _tokenizer

# def ocr_image(image_path: str):
#     model, tokenizer = get_engine()

#     prompt="<image>\n<|grounding|>Convert the document to markdown."

#     res=model.infer(
#         tokenizer,
#         prompt=prompt,
#         image_file=image_path,
#         output_path=".",
#         base_size=1024,
#         image_size=640,
#         crop_mode=True,
#         save_results=False
#     )
#     return res