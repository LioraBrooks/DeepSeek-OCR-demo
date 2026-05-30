import sys
import os
#获取当前脚本的绝对路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#获取项目根目录
PROJECT_ROOT=os.path.dirname(CURRENT_DIR)
#获取backend_ds的绝对路径
BACKEND_DS_DIR=os.path.join(PROJECT_ROOT,"backend_ds")
#添加backend_ds目录到系统路径
if BACKEND_DS_DIR not in sys.path:
    sys.path.append(BACKEND_DS_DIR)

import uuid
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from ocr_engine import get_engine
from fastapi import Response
from evaluator.evaluate import evaluate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

# 前端静态页
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.head("/")
def head_root():
    return Response(status_code=200)

@app.get("/", response_class=HTMLResponse)
def home():
    with open(os.path.join(BASE_DIR, "static", "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/ocr")
async def api_ocr(file: UploadFile = File(...)):
    try:
        ext = os.path.splitext(file.filename)[-1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            return {"ok": False, "error": "只支持 jpg/jpeg/png/webp/bmp"}

        name = f"{uuid.uuid4().hex}{ext}"
        path = os.path.join(UPLOAD_DIR, name)

        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        #1.执行识别
        text = get_engine().ocr(path)

        #2.将识别的结果保存为临时txt以供评估使用
        temp_ocr_txt=path.replace(os.path.splitext(path)[-1], ".txt")
        with open(temp_ocr_txt, "w", encoding="utf-8") as f:
            f.write(text)

        #3.自动执行精度评估
        try:
            gt_path=os.path.join(PROJECT_ROOT,"ground_truth","full.docx")
            eval_result=evaluate(os.path.abspath(temp_ocr_txt),gt_path)

            #调试
            print("OCR TEXT:",text)
            print("EVAL:",eval_result)

        #将识别结果与评估精度一起返回给前端
            return {
                "ok": True, 
                "text": text,
                "accuracy": eval_result["similarity"],  #相似度
                "error_rate": eval_result["error_rate"],  #错误率
                }
        
        
        except Exception as e:
            print(f"评估失败：{e}")
            return {"ok": True, "text": text, "accuracy": 0.0, "error_rate": 1.0}

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"模型调用异常：{e}")
        return {"ok": False, "error": str(e)}
