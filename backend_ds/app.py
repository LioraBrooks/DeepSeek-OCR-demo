#print("FAST API SERVER START")
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from ocr_service import ocr_service
#from ocr_engine import ocr_engine
from evaluator.evaluate import evaluate
import glob

app = FastAPI()

#获取backend_ds的绝对路径
CURRENT_DIR=os.path.dirname(os.path.abspath(__file__))
#获取项目根目录(deepseek_ocr_demo)
PROJECT_ROOT=os.path.dirname(CURRENT_DIR)
UPLOAD_DIR = os.path.join(CURRENT_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_image_name_from_history_path(history_path):
    name = os.path.splitext(os.path.basename(history_path))[0]
    image_name, _, suffix = name.rpartition("_")
    if image_name and len(suffix) == 6:
        return image_name
    return name


def resolve_ground_truth_path(history_path):
    ground_truth_dir = os.path.join(PROJECT_ROOT, "ground_truth")
    image_name = get_image_name_from_history_path(history_path)
    page_gt_path = os.path.join(ground_truth_dir, f"{image_name}.docx")

    if os.path.exists(page_gt_path):
        return page_gt_path, False

    return os.path.join(ground_truth_dir, "full.docx"), True


@app.post("/ocr")
async def ocr_api(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
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
    except Exception as e:
        import traceback
        print("OCR API 调用失败")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "file": file.filename
            }
        )


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

        #3.优先按图片名匹配分页 ground truth，找不到再回退到 full.docx
        gt_path, use_alignment = resolve_ground_truth_path(latest_ocr_path)

        print(f"正在读取:{gt_path}")

        #print(f"GT绝对路径为:{os.path.abspath(gt_path)}")

        #执行评估
        result = evaluate(latest_ocr_path, gt_path, use_alignment=use_alignment)

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
