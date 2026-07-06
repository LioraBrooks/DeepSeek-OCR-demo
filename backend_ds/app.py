from concurrent.futures import ThreadPoolExecutor
import glob
import os
import shutil
import time
import uuid

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from evaluator.evaluate import evaluate
from ocr_service import ocr_service


app = FastAPI()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
UPLOAD_DIR = os.path.join(CURRENT_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ocr_executor = ThreadPoolExecutor(max_workers=1)
ocr_jobs = {}


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


def run_ocr_job(job_id, file_path, filename):
    ocr_jobs[job_id]["status"] = "running"
    ocr_jobs[job_id]["started_at"] = time.time()
    try:
        result = ocr_service.recognize(file_path)
        ocr_jobs[job_id].update({
            "status": "done",
            "text": result,
            "finished_at": time.time()
        })
    except Exception as e:
        import traceback
        print("OCR background job failed")
        traceback.print_exc()
        ocr_jobs[job_id].update({
            "status": "error",
            "error": str(e),
            "file": filename,
            "finished_at": time.time()
        })


@app.post("/ocr")
async def ocr_api(file: UploadFile = File(...)):
    try:
        job_id = uuid.uuid4().hex
        job_dir = os.path.join(UPLOAD_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        file_path = os.path.join(job_dir, file.filename)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        ocr_jobs[job_id] = {
            "status": "queued",
            "file": file.filename,
            "text": "",
            "queued_at": time.time()
        }
        ocr_executor.submit(run_ocr_job, job_id, file_path, file.filename)

        return {
            "job_id": job_id,
            "status": "queued"
        }
    except Exception as e:
        import traceback
        print("OCR API failed")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "file": file.filename
            }
        )


@app.get("/ocr/status/{job_id}")
def ocr_status(job_id: str):
    job = ocr_jobs.get(job_id)
    if not job:
        return JSONResponse(
            status_code=404,
            content={"error": "OCR job not found"}
        )
    result = dict(job)
    start_time = result.get("started_at") or result.get("queued_at")
    end_time = result.get("finished_at") or time.time()
    if start_time:
        result["elapsed_seconds"] = round(end_time - start_time, 1)
    return result


@app.post("/evaluate")
def evaluate_api():
    try:
        history_dir = os.path.join(CURRENT_DIR, "outputs", "history")
        history_files = glob.glob(os.path.join(history_dir, "*.txt"))

        if not history_files:
            return {
                "error": "未找到可评估的识别历史文件"
            }
        latest_ocr_path = max(history_files, key=os.path.getctime)
        gt_path, use_alignment = resolve_ground_truth_path(latest_ocr_path)

        print(f"正在读取:{gt_path}")
        result = evaluate(latest_ocr_path, gt_path, use_alignment=use_alignment)

        return result

    except Exception as e:
        import traceback
        print("evaluate API failed")
        traceback.print_exc()
        return {
            "error": str(e)
        }


@app.get("/ping")
def ping():
    print("PING HIT")
    return {"msg": "ok"}
