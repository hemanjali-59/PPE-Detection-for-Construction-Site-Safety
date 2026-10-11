"""Full local Flask interface for the five trained PPE detectors."""

from pathlib import Path
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

PROJECT_DIR = Path(__file__).resolve().parent
DATA_YAML = PROJECT_DIR / "data" / "data.yaml"
METRICS_FILE = PROJECT_DIR / "results" / "comparison" / "five_model_comparison.csv"

MODEL_CONFIG = {
    "YOLOv8n": {
        "weights": PROJECT_DIR / "models" / "yolov8n.pt",
        "family": "YOLOv8",
        "size": "nano",
    },
    "YOLOv10n": {
        "weights": PROJECT_DIR / "models" / "yolov10n.pt",
        "family": "YOLOv10",
        "size": "nano",
    },
    "YOLO11n": {
        "weights": PROJECT_DIR / "models" / "yolo11n.pt",
        "family": "YOLO11",
        "size": "nano",
    },
    "YOLOv8s": {
        "weights": PROJECT_DIR / "models" / "yolov8s.pt",
        "family": "YOLOv8",
        "size": "small",
    },
    "YOLO11s": {
        "weights": PROJECT_DIR / "models" / "yolo11s.pt",
        "family": "YOLO11",
        "size": "small",
    },
}
MODEL_ORDER = list(MODEL_CONFIG)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
UPLOAD_DIR = PROJECT_DIR / "static" / "uploads"
RESULT_DIR = PROJECT_DIR / "static" / "results"
PLOT_DIR = PROJECT_DIR / "static" / "plots"
for folder in (UPLOAD_DIR, RESULT_DIR, PLOT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
MODEL_CACHE = {}


def model_status():
    """Return the model information displayed on the upload page."""
    rows = []
    for name, details in MODEL_CONFIG.items():
        path = details["weights"]
        rows.append(
            {
                "name": name,
                "family": details["family"],
                "size": details["size"],
                "available": path.is_file(),
                "size_mb": path.stat().st_size / (1024 ** 2) if path.is_file() else None,
            }
        )
    return rows


def get_model(model_name):
    """Load the selected trained model once and reuse it."""
    if model_name not in MODEL_CONFIG:
        raise ValueError("Please select one of the five available models.")
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    weights_path = MODEL_CONFIG[model_name]["weights"]
    if not weights_path.is_file():
        raise FileNotFoundError(f"The trained weight file is missing: {weights_path.name}")

    from ultralytics import YOLO

    MODEL_CACHE[model_name] = YOLO(str(weights_path))
    return MODEL_CACHE[model_name]


def get_metrics(model_name):
    """Read the selected model's genuine saved test metrics."""
    comparison = pd.read_csv(METRICS_FILE)
    row = comparison[comparison["Model"] == model_name]
    if row.empty:
        return {
            "Precision": None,
            "Recall": None,
            "F1": None,
            "mAP50": None,
            "mAP50-95": None,
        }

    values = row.iloc[0]
    return {
        "Precision": float(values["Precision"]),
        "Recall": float(values["Recall"]),
        "F1": float(values["F1"]),
        "mAP50": float(values["mAP@50"]),
        "mAP50-95": float(values["mAP@50-95"]),
    }


def normalize_class_name(name):
    return str(name).strip().lower().replace("_", "-").replace(" ", "-")


def is_violation_class(name):
    return normalize_class_name(name).startswith("no-")


def analyse_detections(detections):
    """Summarize people and explicit negative PPE classes."""
    violation_classes = sorted(
        {item["class_name"] for item in detections if is_violation_class(item["class_name"])}
    )
    violations = [
        f"Missing {normalize_class_name(name)[3:].replace('-', ' ').title()}"
        for name in violation_classes
    ]
    return {
        "status": "VIOLATION" if violations else "SAFE",
        "violations": violations,
        "person_count": sum(
            normalize_class_name(item["class_name"]) == "person" for item in detections
        ),
    }


def save_confidence_chart(detections, output_path):
    """Save a colorful per-object confidence chart."""
    labels = []
    occurrences = {}
    for item in detections:
        name = item["class_name"]
        occurrences[name] = occurrences.get(name, 0) + 1
        labels.append(name if occurrences[name] == 1 else f"{name} #{occurrences[name]}")

    scores = [item["confidence"] for item in detections]
    height = max(3.2, min(8.0, 0.45 * max(len(labels), 1) + 1.8))
    figure, axis = plt.subplots(figsize=(8, height))
    if labels:
        colors = [
            "#dc2626" if is_violation_class(item["class_name"]) else "#0f766e"
            for item in detections
        ]
        bars = axis.barh(labels, scores, color=colors)
        axis.bar_label(bars, labels=[f"{score:.1%}" for score in scores], padding=4)
        axis.set_xlim(0, 1.08)
        axis.invert_yaxis()
    else:
        axis.text(
            0.5,
            0.5,
            "No detections above the selected threshold",
            ha="center",
            va="center",
        )
        axis.set_xticks([])
        axis.set_yticks([])
    axis.set_xlabel("Detection confidence")
    axis.set_title("Detected Classes and Confidence Scores")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def run_inference(model_name, image_path, confidence):
    model = get_model(model_name)
    result = model.predict(
        source=str(image_path),
        conf=confidence,
        imgsz=640,
        verbose=False,
    )[0]

    class_ids = result.boxes.cls.detach().cpu().int().tolist()
    scores = result.boxes.conf.detach().cpu().tolist()
    result_names = result.names if isinstance(result.names, dict) else dict(enumerate(result.names))
    detections = [
        {
            "class_name": str(result_names.get(class_id, f"Class {class_id}")),
            "confidence": float(score),
        }
        for class_id, score in zip(class_ids, scores)
    ]
    return result, detections


@app.route("/", methods=["GET", "POST"])
def index():
    selected_model = request.form.get("model", "YOLO11n")
    confidence = request.form.get("confidence", "0.25")

    if request.method == "POST":
        try:
            if selected_model not in MODEL_ORDER:
                raise ValueError("Please select one of the five available models.")
            try:
                confidence_value = float(confidence)
            except ValueError as error:
                raise ValueError("Confidence must be a number between 0.05 and 0.95.") from error
            if not 0.05 <= confidence_value <= 0.95:
                raise ValueError("Confidence must be between 0.05 and 0.95.")

            uploaded = request.files.get("image")
            if not uploaded or not uploaded.filename:
                raise ValueError("Please choose a JPG, JPEG, or PNG image.")
            if Path(uploaded.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
                raise ValueError("Unsupported file type. Please upload JPG, JPEG, or PNG.")

            safe_stem = Path(secure_filename(uploaded.filename)).stem or "upload"
            unique_name = f"{uuid4().hex}_{safe_stem}.jpg"
            upload_path = UPLOAD_DIR / unique_name
            try:
                Image.open(uploaded.stream).convert("RGB").save(
                    upload_path,
                    format="JPEG",
                    quality=95,
                )
            except (UnidentifiedImageError, OSError) as error:
                raise ValueError("The uploaded file is not a readable image.") from error

            result, detections = run_inference(selected_model, upload_path, confidence_value)

            annotated_name = f"annotated_{unique_name}"
            annotated_path = RESULT_DIR / annotated_name
            annotated_bgr = result.plot()
            Image.fromarray(annotated_bgr[:,:,::-1]).save(annotated_path, quality=95)

            chart_name = f"confidence_{Path(unique_name).stem}.png"
            save_confidence_chart(detections, PLOT_DIR / chart_name)

            return render_template(
                "result.html",
                selected_model=selected_model,
                confidence_threshold=confidence_value,
                detections=detections,
                compliance=analyse_detections(detections),
                metrics=get_metrics(selected_model),
                original_url=url_for("static", filename=f"uploads/{unique_name}"),
                annotated_url=url_for("static", filename=f"results/{annotated_name}"),
                chart_url=url_for("static", filename=f"plots/{chart_name}"),
            )
        except Exception as error:
            return render_template(
                "index.html",
                models=model_status(),
                selected_model=selected_model,
                confidence=confidence,
                error=str(error),
            ), 400

    return render_template(
        "index.html",
        models=model_status(),
        selected_model=selected_model,
        confidence=confidence,
        error=None,
    )


@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "index.html",
        models=model_status(),
        selected_model="YOLO11n",
        confidence="0.25",
        error="The image is larger than the 10 MB upload limit.",
    ), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
