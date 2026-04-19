import torch
from transformers import AutoModel, AutoTokenizer
import os
import shutil
import uuid
import glob

MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "DeepSeek-OCR-model")
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DeepSeekOCRService:
    def __init__(self):
        print("Loading DeepSeek-OCR model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            local_files_only=True
        )

        self.model = AutoModel.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            local_files_only=True,
            use_safetensors=True
        )

        self.model = self.model.eval().to(DEVICE)

        if DEVICE == "cuda":
            self.model = self.model.to(torch.bfloat16)

        print("DeepSeek-OCR model loaded.")

    def recognize(self, image_path: str):
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "

        #output_dir = "outputs"
        #os.makedirs(output_dir, exist_ok=True)

        #output_path = os.path.join(output_dir, str(uuid.uuid4()))

        #调用模型
        self.model.infer(
            self.tokenizer,
            prompt=prompt,
            image_file=image_path,
            #output_path=output_path,            
            output_path="outputs/text",         #固定路径方便观察

            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=True,
            test_compress=True
        )

        #全局查找mmd文件
        mmd_files=glob.glob("outputs/**/*.mmd",recursive=True)
        #print("mmd_files=",mmd_files)

        if not mmd_files:
            print("WARNING: 没找到.mmd文件")
            return ""


        #取最新的mmd文件
        latest_file=max(mmd_files, key=os.path.getctime)
        print("使用mmd文件:",latest_file)

        #读取原始内容
        with open(latest_file, "r", encoding="utf-8") as f:
            raw_text = f.read()

        #保存原始识别结果--用于评估
        image_name=os.path.splitext(os.path.basename(image_path))[0]

        os.makedirs("outputs/history",exist_ok=True)
        save_path=f"outputs/history/{image_name}_{uuid.uuid4().hex[:6]}.mmd"
        

        with open(save_path,"w",encoding="utf-8") as f:
            f.write(raw_text)


        #===== 清洗输出 =====
        lines = raw_text.split("\n")
        clean_lines = []

        for line in lines:
            line = line.strip()

            #过滤空行
            if not line:
                continue

            #过滤对话残留
            if line.lower().startswith(("system", "user", "assistant")):
                continue
            #过滤markdown图片/标签
            if line.startswith("![]") or line.startswith("<"):
                continue
            #去掉多与markdown符号
            line=line.replace("**","").replace("#","")

            clean_lines.append(line)

        

        #去重复
        clean_lines_unique=[]
        prev_line=""

        for line in clean_lines:
            if line != prev_line:
                clean_lines_unique.append(line)
                prev_line = line

        clean_text = "\n".join(clean_lines_unique)

        #保存txt
        save_txt_path=save_path.replace(".mmd",".txt")
        with open(save_txt_path,"w",encoding="utf-8") as f:
            f.write(clean_text)
            
        #返回结果
        return clean_text
                

        #调试
        #print("DEBUG res=",res,type(res))


ocr_service = DeepSeekOCRService()