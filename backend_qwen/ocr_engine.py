import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen2VLForConditionalGeneration
import threading
import re

try:
    _HAS_QWEN2VL = True
except Exception:
    _HAS_QWEN2VL = False
#动态获取模型路径
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ID=os.path.join(BASE_DIR,"models","Qwen2-VL-2B-Instruct")
#MODEL_ID = "/home/lxr/projects/deepseek_ocr_demo/models/Qwen2-VL-2B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_lock = threading.Lock()#全局锁

def shrink(img:Image.Image, max_side=1280)->Image.Image:  #缩放函数
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / m
    return img.resize((int(w*scale),int(h*scale)))

class OCREngine:
    def __init__(self):
        print(f">>>>正在从{MODEL_ID}加载模型...")

        if not _HAS_QWEN2VL:
            raise RuntimeError("你的 transformers 版本不支持 Qwen2VLForConditionalGeneration。请升级 transformers 到较新版本。")

        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True,local_files_only=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            device_map="auto",
            #use_safetensors=True,
            #trust_remote_code=True,
            local_files_only=True
        )
        self.model.eval()

        print(">>>>模型加载完毕",flush=True)

    @staticmethod
    def clean_qwen_output(text):
              
        text = re.sub(r'<[^>]+>', '', text)      
        
        text=text.replace("assistant","")

        return text.strip()

    @torch.inference_mode()
    def ocr(self, image_path: str) -> str:
        with _lock:
            image = Image.open(image_path).convert("RGB")
            image=shrink(image)


        # prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Extract all text from the image.Only output text."},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text_prompt], images=[image], padding=True,return_tensors="pt")

        if DEVICE == "cuda":
            inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            #use_cache=True
            #top_k=None,
            #top_p=None
        )

        input_len=inputs["input_ids"].shape[1]
        decoded = self.processor.batch_decode(out_ids[:, input_len:], skip_special_tokens=True)[0]

        return self.clean_qwen_output(decoded)
        
        

_ocr_engine =None
def get_engine():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine=OCREngine()
    return _ocr_engine
