import base64
import io
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import requests
from flask import Flask, render_template, request
from PIL import Image

from image_processing import count_chickpeas_advanced


os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
# Avoid GPU/CUDA crashes on some Windows setups by defaulting to CPU inference.
# (You can remove this if you have a stable CUDA stack and want GPU.)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# Reduce native thread fan-out (can cause hard crashes in some Windows stacks).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


APP_ROOT = Path(__file__).resolve().parent
QUALITY_MODEL_DIR = APP_ROOT / "trained_model"
COUNT_MODEL_DIR = APP_ROOT / "count_model"

app = Flask(
    __name__,
    root_path=str(APP_ROOT),
    template_folder=str(APP_ROOT / "templates"),
    static_folder=str(APP_ROOT / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15MB

_QUALITY_CACHE: tuple | None = None
_COUNT_CACHE: tuple | None = None


def _pil_from_upload_or_url(file_storage, image_url: str | None) -> Image.Image | None:
    if image_url:
        r = requests.get(image_url, timeout=15)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content))

    if file_storage and file_storage.filename:
        return Image.open(file_storage.stream)

    return None


def _bgr_to_data_url_png(bgr_img: np.ndarray) -> str:
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode annotated image as PNG.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _rgb_to_data_url_png(rgb_img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode image as PNG.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _pil_to_data_url_png(pil_img: Image.Image) -> str:
    out = io.BytesIO()
    pil_img.convert("RGB").save(out, format="PNG")
    b64 = base64.b64encode(out.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def extract_seed_features_from_contours(pil_image: Image.Image, contours) -> list[dict]:
    """Lightweight per-contour features for UI listing/graphs."""
    if not contours:
        return []
    feats: list[dict] = []
    for i, cnt in enumerate(contours, start=1):
        try:
            area = float(cv2.contourArea(cnt))
            if area <= 0:
                continue
            perimeter = float(cv2.arcLength(cnt, True))
            circ = float((4.0 * np.pi * area) / (perimeter * perimeter + 1e-6)) if perimeter > 0 else 0.0
            x, y, w, h = cv2.boundingRect(cnt)
            eq_d = float(np.sqrt(4.0 * area / np.pi))
            feats.append(
                {
                    "id": i,
                    "area_px": area,
                    "perimeter_px": perimeter,
                    "circularity": circ,
                    "equivalent_diameter_px": eq_d,
                    "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                }
            )
        except Exception:
            continue
    return feats


def load_quality_model(model_dir: Path):
    global _QUALITY_CACHE
    if _QUALITY_CACHE is not None:
        return _QUALITY_CACHE
    if not model_dir.exists():
        return None, None, None
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        processor = AutoImageProcessor.from_pretrained(str(model_dir))
        model = AutoModelForImageClassification.from_pretrained(str(model_dir))
        model.to(torch.device("cpu"))
        model.eval()
        class_names_path = model_dir / "class_names.json"
        if class_names_path.exists():
            import json

            class_names = json.loads(class_names_path.read_text(encoding="utf-8"))
        else:
            class_names = ["healthy", "broken", "discolored"]
        _QUALITY_CACHE = (processor, model, class_names)
        return _QUALITY_CACHE
    except Exception:
        return None, None, None


def enhanced_preprocess_pil(image: Image.Image, canny_low: int = 80, canny_high: int = 180) -> Image.Image:
    arr = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v2 = clahe.apply(v)
    bgr_clahe = cv2.cvtColor(cv2.merge((h, s, v2)), cv2.COLOR_HSV2BGR)

    gray = cv2.cvtColor(bgr_clahe, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)
    edges_col = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined = cv2.addWeighted(bgr_clahe, 0.85, edges_col, 0.15, 0)
    rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def predict_quality(pil_image: Image.Image, canny_low: int, canny_high: int):
    processor, model, class_names = load_quality_model(QUALITY_MODEL_DIR)
    if not (processor and model and class_names):
        return None

    try:
        import torch

        processed = enhanced_preprocess_pil(pil_image, canny_low=canny_low, canny_high=canny_high)
        inputs = processor(images=processed, return_tensors="pt")
        inputs = {k: v.to(torch.device("cpu")) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        return {
            "predicted_class": str(class_names[idx]),
            "confidence": float(probs[idx]),
            "all_predictions": {str(class_names[i]): float(probs[i]) for i in range(len(class_names))},
        }
    except Exception:
        return None


def load_count_model():
    global _COUNT_CACHE
    if _COUNT_CACHE is not None:
        return _COUNT_CACHE
    model_path = COUNT_MODEL_DIR / "model.safetensors"
    if not model_path.exists():
        return None, None
    try:
        import torch
        import torch.nn as nn
        from safetensors.torch import load_file as load_safetensors
        from transformers import AutoImageProcessor, ViTModel

        class CountRegressor(nn.Module):
            def __init__(self, backbone_name: str = "google/vit-base-patch16-224"):
                super().__init__()
                self.backbone = ViTModel.from_pretrained(backbone_name)
                hidden = self.backbone.config.hidden_size
                self.regressor = nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, 256),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 1),
                )

            def forward(self, pixel_values):
                out = self.backbone(pixel_values=pixel_values)
                return self.regressor(out.pooler_output)

        processor = AutoImageProcessor.from_pretrained(str(COUNT_MODEL_DIR))
        model = CountRegressor()
        model.load_state_dict(load_safetensors(str(model_path)))
        model.to(torch.device("cpu"))
        model.eval()
        _COUNT_CACHE = (processor, model)
        return _COUNT_CACHE
    except Exception:
        return None, None


def preprocess_edges_for_processor(pil_image: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    arr = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v2 = clahe.apply(v)
    bgr = cv2.cvtColor(cv2.merge((h, s, v2)), cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, low, high)
    edges3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges3)


def predict_count_ml(pil_image: Image.Image):
    processor, model = load_count_model()
    if not (processor and model):
        return None
    try:
        import torch

        img_edges = preprocess_edges_for_processor(pil_image)
        inputs = processor(images=img_edges, return_tensors="pt")
        inputs = {k: v.to(torch.device("cpu")) for k, v in inputs.items()}
        with torch.no_grad():
            pred = model(pixel_values=inputs["pixel_values"])
        log_count = float(pred.squeeze().item())
        return float(max(0.0, np.expm1(log_count)))
    except Exception:
        return None


@app.get("/")
def index():
    return render_template(
        "index.html",
        result=None,
        model_present=QUALITY_MODEL_DIR.exists(),
        count_model_present=(COUNT_MODEL_DIR / "model.safetensors").exists(),
    )


@app.post("/analyze")
def analyze():
    image_url = (request.form.get("image_url") or "").strip()
    use_ml_count = (request.form.get("use_ml_count") == "on")
    try:
        min_area = int(request.form.get("min_area") or 10)
        max_area = int(request.form.get("max_area") or 10000)
        canny_low = int(request.form.get("canny_low") or 80)
        canny_high = int(request.form.get("canny_high") or 180)
    except ValueError:
        return render_template("index.html", result={"error": "Invalid numeric parameter."})

    pil_img = None
    try:
        pil_img = _pil_from_upload_or_url(request.files.get("image_file"), image_url or None)
        if pil_img is None:
            return render_template("index.html", result={"error": "Please upload an image or provide an image URL."})
        pil_img = pil_img.convert("RGB")
    except Exception as e:
        return render_template("index.html", result={"error": f"Failed to load image: {e}"})

    # Count via contours: function expects a file path for consistent OpenCV behavior.
    contour_result = None
    tmp_path = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_path = tmp.name
        tmp.close()
        pil_img.save(tmp_path, format="PNG")
        contour_result = count_chickpeas_advanced(tmp_path, min_area=min_area, max_area=max_area)
    except Exception as e:
        contour_result = {"error": f"Contour counting failed: {e}"}
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    ml_count = predict_count_ml(pil_img) if use_ml_count else None

    final_count = None
    contour_count = None
    if contour_result and "count" in contour_result and contour_result.get("error") is None:
        contour_count = contour_result.get("count")

    if ml_count is not None and contour_count is not None:
        weight_ml = 0.6 if float(contour_count) > 100 else 0.35
        final_count = float(round(weight_ml * float(ml_count) + (1 - weight_ml) * float(contour_count), 2))
    elif ml_count is not None:
        final_count = float(round(float(ml_count), 2))
    elif contour_count is not None:
        final_count = float(contour_count)

    quality = predict_quality(pil_img, canny_low=canny_low, canny_high=canny_high)

    annotated_url = None
    try:
        if contour_result and isinstance(contour_result, dict) and contour_result.get("annotated_image") is not None:
            annotated_url = _bgr_to_data_url_png(contour_result["annotated_image"])
    except Exception:
        annotated_url = None

    seed_features = []
    try:
        if isinstance(contour_result, dict) and contour_result.get("contours"):
            seed_features = extract_seed_features_from_contours(pil_img, contour_result["contours"])
    except Exception:
        seed_features = []

    # Precompute arrays for lightweight JS charts
    chart_data = None
    if seed_features:
        areas = [float(f["area_px"]) for f in seed_features]
        circs = [float(f["circularity"]) for f in seed_features]
        chart_data = {
            "areas": areas,
            "circularities": circs,
        }

    result = {
        "original_image_url": _pil_to_data_url_png(pil_img),
        "annotated_image_url": annotated_url,
        "quality": quality,
        "contour_count": contour_count,
        "ml_count": (float(round(ml_count, 2)) if ml_count is not None else None),
        "final_count": final_count,
        "analysis": (contour_result.get("analysis") if isinstance(contour_result, dict) else None),
        "seed_features": seed_features,
        "chart_data": chart_data,
        "error": (contour_result.get("error") if isinstance(contour_result, dict) else None),
    }

    return render_template(
        "index.html",
        result=result,
        model_present=QUALITY_MODEL_DIR.exists(),
        count_model_present=(COUNT_MODEL_DIR / "model.safetensors").exists(),
    )


if __name__ == "__main__":
    # On some Windows setups, the debug reloader can get into a reload loop by
    # watching files under site-packages (e.g. transformers), which looks like
    # ERR_CONNECTION_RESET in the browser. Run without the reloader for stability.
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=False)

