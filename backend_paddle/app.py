from concurrent.futures import ThreadPoolExecutor
import glob
import os
import shutil
import sys
import time
import uuid

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ocr_engine import get_engine


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend_ds.evaluator.evaluate import evaluate  # noqa: E402


UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
HISTORY_DIR = os.path.join(OUTPUT_DIR, "history")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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


def save_history_text(image_path, text):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(HISTORY_DIR, f"{image_name}_{uuid.uuid4().hex[:6]}.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    return save_path


def run_ocr_job(job_id, file_path, filename):
    ocr_jobs[job_id]["status"] = "running"
    ocr_jobs[job_id]["started_at"] = time.time()
    try:
        text = get_engine().ocr_image(file_path)
        history_path = save_history_text(file_path, text)
        ocr_jobs[job_id].update({
            "status": "done",
            "text": text,
            "history_path": history_path,
            "finished_at": time.time()
        })
    except Exception as e:
        import traceback
        print("Paddle OCR background job failed")
        traceback.print_exc()
        ocr_jobs[job_id].update({
            "status": "error",
            "error": str(e),
            "file": filename,
            "finished_at": time.time()
        })


@app.get("/", response_class=HTMLResponse)
def home():
    with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.head("/")
def head_root():
    return Response(status_code=200)


@app.post("/ocr")
@app.post("/api/ocr")
async def api_ocr(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        return JSONResponse(
            status_code=400,
            content={"error": "只支持 jpg/jpeg/png/bmp/webp 图片"}
        )

    try:
        job_id = uuid.uuid4().hex
        job_dir = os.path.join(UPLOAD_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        path = os.path.join(job_dir, file.filename)

        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        ocr_jobs[job_id] = {
            "status": "queued",
            "file": file.filename,
            "text": "",
            "queued_at": time.time()
        }
        ocr_executor.submit(run_ocr_job, job_id, path, file.filename)

        return {
            "ok": True,
            "job_id": job_id,
            "status": "queued"
        }
    except Exception as e:
        import traceback
        print("Paddle OCR API failed")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "error_type": type(e).__name__,
                "file": file.filename
            }
        )


@app.get("/ocr/status/{job_id}")
def ocr_status(job_id: str):
    job = ocr_jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "OCR job not found"})

    result = dict(job)
    start_time = result.get("started_at") or result.get("queued_at")
    end_time = result.get("finished_at") or time.time()
    if start_time:
        result["elapsed_seconds"] = round(end_time - start_time, 1)
    return result


@app.post("/evaluate")
def evaluate_api():
    try:
        history_files = glob.glob(os.path.join(HISTORY_DIR, "*.txt"))

        if not history_files:
            return {"error": "未找到可评估的识别历史文件"}

        latest_ocr_path = max(history_files, key=os.path.getctime)
        gt_path, use_alignment = resolve_ground_truth_path(latest_ocr_path)
        result = evaluate(latest_ocr_path, gt_path, use_alignment=use_alignment)
        result["engine"] = "paddleocr"

        return result
    except Exception as e:
        import traceback
        print("Paddle evaluate API failed")
        traceback.print_exc()
        return {"error": str(e), "error_type": type(e).__name__}


@app.get("/ping")
def ping():
    return {"msg": "ok", "engine": "paddleocr"}
