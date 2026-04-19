#print("FAST API SERVER START")
from fastapi import FastAPI, UploadFile, File
import shutil
import os
from ocr_service import ocr_service
#from ocr_engine import ocr_engine
from evaluator.evaluate import evaluate
import glob

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

#获取backend_ds的绝对路径
CURRENT_DIR=os.path.dirname(os.path.abspath(__file__))
#获取项目根目录(deepseek_ocr_demo)
PROJECT_ROOT=os.path.dirname(CURRENT_DIR)
@app.post("/ocr")
async def ocr_api(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = ocr_service.recognize(file_path)

    #调试信息
    #print(f"识别结果: {result}")
    #print("API RETURN:",result)

    #text=ocr_engine(path)

    return {
        "text": result
    }


@app.post("/evaluate")
def evaluate_api():
    try:
        #print("evaluate API 进入调用")

        #1.OCR识别结果在backend_ds/outputs/history目录下
        history_dir=os.path.join(CURRENT_DIR,"outputs","history")

        #2.找到最新的识别历史文件
        history_files=glob.glob(os.path.join(history_dir, "*.txt"))

        if not history_files:
            return {
                "error":"未找到可评估的识别历史文件"
            }
        latest_ocr_path=max(history_files, key=os.path.getctime)

        #3.定位到统计目录下的ground_truth
        
        gt_path=os.path.join(PROJECT_ROOT,"ground_truth","full.docx")

        print(f"正在读取:{gt_path}")

        #print(f"GT绝对路径为:{os.path.abspath(gt_path)}")

        #执行评估
        result = evaluate(latest_ocr_path, gt_path)

        #print("评估结果:",result)

        #print("evaluate API 被调用成功")

        return result

    except Exception as e:
        import traceback
        print("evaluate API 调用失败")
        traceback.print_exc()
        return {
            "error": str(e)
        }

@app.get("/ping")
def ping():
    print("PING HIT")
    return {"msg": "ok"}