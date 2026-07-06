import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"]="True"
os.environ["DISABLE_AUTO_LOGGING_CONFIG"]="1"
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_onednn", "0")
from paddleocr import PaddleOCR
from PIL import Image
import threading
import time
import inspect

_lock = threading.Lock()

def shrink(img: Image.Image, max_side: int = 1600) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale=max_side / m
    return img.resize((int(w*scale),int(h*scale)))

class OCREngine:
    def __init__(self):
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        os.environ.setdefault("FLAGS_enable_pir_api", "0")
        os.environ.setdefault("FLAGS_enable_onednn", "0")
        self.api_mode = os.getenv("PADDLE_OCR_API", "auto")
        det_model_name = os.getenv("PADDLE_DET_MODEL", "PP-OCRv4_mobile_det")
        rec_model_name = os.getenv("PADDLE_REC_MODEL", "PP-OCRv4_mobile_rec")
        det_model_dir = os.getenv("PADDLE_DET_MODEL_DIR") or None
        rec_model_dir = os.getenv("PADDLE_REC_MODEL_DIR") or None
        device = os.getenv("PADDLE_DEVICE", "cpu")

        kwargs = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "text_detection_model_name": det_model_name,
            "text_recognition_model_name": rec_model_name,
            "device": device,
            "use_angle_cls": False,
            "lang": os.getenv("PADDLE_LANG", "ch"),
            "use_gpu": device.startswith("gpu"),
            "show_log": False
        }
        if os.getenv("PADDLE_ENABLE_MKLDNN", "false").lower() not in ("1", "true", "yes", "on"):
            kwargs["enable_mkldnn"] = False
        if det_model_dir:
            kwargs["text_detection_model_dir"] = det_model_dir
        if rec_model_dir:
            kwargs["text_recognition_model_dir"] = rec_model_dir

        print(
            "PaddleOCR config:",
            f"det={det_model_name}",
            f"rec={rec_model_name}",
            f"det_dir={det_model_dir}",
            f"rec_dir={rec_model_dir}",
            f"device={device}",
            f"api={self.api_mode}"
        )
        start = time.time()
        supported_params = inspect.signature(PaddleOCR).parameters
        kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
        self.ocr = PaddleOCR(**kwargs)
        print(f"PaddleOCR model loaded in {time.time() - start:.1f}s")

    def ocr_image(self, image_path: str) -> str:
        with _lock:
            start = time.time()
            img = Image.open(image_path).convert("RGB")
            img = shrink(img, max_side=1600)
            temp_path = image_path + ".tmp.jpg"
            img.save(temp_path)

            if self.api_mode == "legacy" or not hasattr(self.ocr, "predict"):
                results = self.ocr.ocr(temp_path, cls=False)
            else:
                results = self.ocr.predict(input=temp_path)
            texts = self._extract_texts(results)
            text = "\n".join(t.strip() for t in texts if t and str(t).strip())
            print(f"PaddleOCR inference finished in {time.time() - start:.1f}s, chars={len(text)}")

            if not text.strip():
                raise RuntimeError("PaddleOCR 没有返回可用识别文本，请检查图片或 PaddleOCR 返回结构")

            return text

    def _extract_texts(self, results):
        texts = []

        for res in results or []:
            data = getattr(res, "json", None)
            if callable(data):
                data = data()
            elif data is None and isinstance(res, dict):
                data = res

            if isinstance(data, dict):
                self._collect_texts_from_dict(data, texts)
            elif isinstance(res, (list, tuple)):
                self._collect_texts_from_list(res, texts)

        return texts

    def _collect_texts_from_dict(self, data, texts):
        for key in ("rec_texts", "res_texts", "texts"):
            value = data.get(key)
            if isinstance(value, list):
                texts.extend(str(v) for v in value if v)

        inner = data.get("res")
        if isinstance(inner, dict):
            self._collect_texts_from_dict(inner, texts)

        rec_text = data.get("rec_text")
        if isinstance(rec_text, list):
            texts.extend(str(v) for v in rec_text if v)
        elif rec_text:
            texts.append(str(rec_text))

    def _collect_texts_from_list(self, data, texts):
        for item in data:
            if isinstance(item, str):
                texts.append(item)
            elif (
                isinstance(item, (list, tuple))
                and len(item) >= 2
                and isinstance(item[1], (list, tuple))
                and item[1]
                and isinstance(item[1][0], str)
            ):
                texts.append(item[1][0])
            elif isinstance(item, dict):
                self._collect_texts_from_dict(item, texts)
            elif isinstance(item, (list, tuple)):
                self._collect_texts_from_list(item, texts)

#暂时改为：(仅测试)
            """
            print("RAW RESULTS:",results)
            for i,res in enumerate(results):
                print(f"--- result {i} ---")
                print("type:",type(res))
                print("dir has json:",hasattr(res,"json"))
                if hasattr(res,"json"):
                    print("json:",res.json)

            return "Done"
            """

            """
            lines=[]
            for res in result:
                # PaddleOCR 3.x 返回对象结构里通常可 print/save，
                # 这里尽量兼容常见结果形式
                if hasattr(res,"json"):
                    data=res.json
                elif isinstance(res,dict):
                    data=res
                else:
                    data=None

                if data and "res_texts" in data:
                    lines.extend(data["rec_texts"])
                elif data and "dt_polys" in data and "rec_text" in data:
                    # 兼容其他结构
                    if isinstance(data["rec_text"], list):
                        lines.extend(data["rec_text"])
                    else:
                        lines.append(str(data["rec_text"]))
                else:
                    # 直接转字符串
                    lines.append(str(res))

            return "\n".join([x.strip() for x in lines if str(x).strip()])
            """

_ocr_engine=None


def get_engine():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine=OCREngine()
    return _ocr_engine
