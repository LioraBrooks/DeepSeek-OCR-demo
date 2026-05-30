import torch
from transformers import AutoModel, AutoTokenizer
import os
import shutil
import uuid
import glob
import inspect

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "DeepSeek-OCR-model")
)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
OCR_MAX_NEW_TOKENS = int(os.getenv("OCR_MAX_NEW_TOKENS", "4096"))

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

    def _infer_with_token_limit(self, **kwargs):
        """DeepSeek-OCR infer hardcodes a large max_new_tokens; clamp it for web requests."""
        original_generate = self.model.generate

        def limited_generate(*args, **generate_kwargs):
            current_limit = generate_kwargs.get("max_new_tokens", OCR_MAX_NEW_TOKENS)
            generate_kwargs["max_new_tokens"] = min(current_limit, OCR_MAX_NEW_TOKENS)
            return original_generate(*args, **generate_kwargs)

        self.model.generate = limited_generate
        try:
            infer_kwargs = dict(kwargs)
            infer_params = inspect.signature(self.model.infer).parameters
            if "eval_mode" in infer_params:
                infer_kwargs["eval_mode"] = True
            if "verbose" in infer_params:
                infer_kwargs["verbose"] = False
            return self.model.infer(**infer_kwargs)
        finally:
            self.model.generate = original_generate

    def recognize(self, image_path: str):
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        request_id = uuid.uuid4().hex
        output_path = os.path.join(OUTPUT_DIR, "text", request_id)

        #output_dir = "outputs"
        #os.makedirs(output_dir, exist_ok=True)

        #output_path = os.path.join(output_dir, str(uuid.uuid4()))

        #调用模型
        infer_result = self._infer_with_token_limit(
            tokenizer=self.tokenizer,
            prompt=prompt,
            image_file=image_path,
            #output_path=output_path,            
            output_path=output_path,

            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=True,
            test_compress=True
        )

        if isinstance(infer_result, str) and infer_result.strip():
            raw_text = infer_result
        else:
            raw_text = ""

        if not raw_text:
            #只读取本次请求的输出，避免失败时误拿历史文件
            mmd_files=glob.glob(os.path.join(output_path, "**", "*.mmd"), recursive=True)
            #print("mmd_files=",mmd_files)

            if not mmd_files:
                raise RuntimeError("OCR 未生成 .mmd 结果文件")


            #取最新的mmd文件
            latest_file=max(mmd_files, key=os.path.getctime)
            print("使用mmd文件:",latest_file)

            #读取原始内容
            with open(latest_file, "r", encoding="utf-8") as f:
                raw_text = f.read()

        #保存原始识别结果--用于评估
        image_name=os.path.splitext(os.path.basename(image_path))[0]

        history_dir = os.path.join(OUTPUT_DIR, "history")
        os.makedirs(history_dir, exist_ok=True)
        save_path=os.path.join(history_dir, f"{image_name}_{uuid.uuid4().hex[:6]}.mmd")
        

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
