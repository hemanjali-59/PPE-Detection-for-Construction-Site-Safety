# PPE Detection for Workplace Safety

A computer vision project that compares five trained YOLO models for detecting personal protective equipment (PPE) and explicit missing-equipment classes. A local Flask application lets users upload an image, select a model, adjust the confidence threshold, and view annotated detections.

## Features

- Compare YOLOv8n, YOLOv10n, YOLO11n, YOLOv8s, and YOLO11s.
- Detect 13 classes covering PPE, missing PPE, and people.
- Review dataset integrity and annotation formats.
- Display bounding boxes, class names, confidence scores, and a confidence chart.
- Flag image-level violations when an explicit NO-PPE class is detected.
- Cache loaded models to avoid repeatedly loading checkpoints.

## Technology Stack

Python, Ultralytics YOLO, PyTorch, Flask, Jinja2, Pandas, Matplotlib, Pillow, PyYAML, and Jupyter Notebook.

## Dataset

The final dataset contains **9,663 images**.

| Split | Images |
| --- | ---: |
| Training | 7,721 |
| Validation | 1,460 |
| Test | 482 |

The final taxonomy contains:

- **PPE:** Boots, Gloves, Goggles, Helmet, Mask, Vest
- **Missing PPE:** NO-Boots, NO-Gloves, NO-Goggles, NO-Helmet, NO-Mask, NO-Vest
- **Worker:** Person

Use the class-ID mapping in the supplied dataset YAML and checkpoints. Do not substitute the earlier 11-class dataset configuration: it belongs to a different project stage.

### Annotation Format

YOLO detection labels use one object per row:

```text
class_id x_center y_center width height
```

Coordinates are normalized relative to the image dimensions. Each image should have a corresponding label file with the same filename stem.

The final data-quality review reported no missing labels, empty label files, or unreadable images. It flagged **109 non-standard annotation rows**: 78 training, 24 validation, and 7 test rows. These rows were retained for review rather than automatically deleted.

### Dataset Paths

Point the YAML `path` to the dataset root directory, **not** to the YAML file itself. For a dataset organized as `images/train`, `images/val`, and `images/test`:

```yaml
path: D:/path/to/dataset
train: images/train
val: images/val
test: images/test
# Retain the supplied 13-class names mapping here.
```

The matching label folders should be `labels/train`, `labels/val`, and `labels/test`. Adjust these paths to the actual dataset layout.

## Main Project Files

The final report identifies these project components; dataset and checkpoint locations must match the paths configured in your local copy.

| Component | Purpose |
| --- | --- |
| `app.py` | Flask application and inference workflow |
| `templates/base.html` | Shared page layout |
| `templates/index.html` | Model selection and image upload |
| `templates/result.html` | Prediction results |
| `PPE_Detection_Refined.ipynb` | Dataset review, model evaluation, and comparisons |
| `dataset/data.yaml` / `data_local.yaml` | Dataset configuration |
| `reports/model_comparison_raw.csv` | Reported test metrics |
| `reports/five_model_comparison.csv` | Consolidated model comparison |
| `training_runs/` and `results/` | Training histories and evaluation artifacts |

Each model requires its trained `best.pt` checkpoint.

## Setup

### 1. Open the Project

Download or clone this repository, then open the project root in VS Code or a terminal.

### 2. Create a Virtual Environment

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

If your project includes `requirements.txt`, use it:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Otherwise, this is a starting dependency set; use the versions compatible with the original checkpoints and notebook:

```bash
python -m pip install ultralytics flask pandas matplotlib pillow pyyaml jupyter
```

### 4. Configure Local Paths

- Set dataset paths in the supplied YAML configuration.
- Set the five checkpoint paths in the application and notebook.
- Ensure the application can locate the saved comparison CSV.
- Keep the dataset taxonomy consistent with the trained checkpoints.

The dataset and model weights may need to be obtained separately if they are not included in the repository.

### 5. Start the Application

From the directory containing `app.py`:

```bash
python -m flask --app app run
```

Open the local address printed in the terminal, normally `http://127.0.0.1:5000`.

## Using the Application

1. Select one of the five YOLO models.
2. Set the confidence threshold; **0.25** is the documented starting value.
3. Upload a JPG, JPEG, or PNG image smaller than 10 MB.
4. Submit the image for inference.
5. Review the annotated image, detection confidences, confidence chart, and PPE status.

The application validates the upload and model selection, runs inference at **640-pixel input size**, and saves the uploaded image, annotated result, and confidence chart.

## Model Evaluation

The final comparison evaluated existing checkpoints without retraining. The refined notebook defaults to `RUN_TRAINING = False`.

| Model | Precision | Recall | F1 | mAP@0.50 | mAP@0.50:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n | 73.92% | 76.09% | 74.99% | 79.19% | 44.40% |
| YOLOv10n | 78.68% | 78.18% | 78.43% | 82.55% | 47.14% |
| YOLO11n | 80.61% | 76.84% | 78.68% | 82.15% | 46.59% |
| **YOLOv8s** | **81.35%** | **80.22%** | **80.78%** | **84.19%** | **50.76%** |
| YOLO11s | 79.65% | 78.35% | 78.99% | 80.63% | 45.85% |

**YOLOv8s achieved the strongest reported performance across all five metrics.**

Evaluation processed **481 of 482 test images**, containing **1,969 labelled instances**. One image was skipped because its annotation mixed detection and segmentation formats.

These are reported results from the final business case study, not newly reproduced benchmarks. They are descriptive results for these checkpoints and this dataset; no statistical significance test was reported.

## PPE Status Logic

Class names are normalized before checking for explicit `NO-` classes.

- **VIOLATION:** At least one explicit missing-PPE class was detected.
- **SAFE:** The existing application label when no explicit missing-PPE class was detected.

Here, **SAFE means only that no explicit violation was detected**. It does not establish that every worker wears all required equipment. Missed detections, occlusion, and low confidence can hide violations. A clearer output label is “No explicit violation detected”; images with no detections should be reported separately.

The current rule operates at the **image level**. It does not associate equipment or violations with individual workers.

## Limitations

- Non-standard annotation rows require normalization and review.
- Class frequencies are uneven.
- Test results are specific to the supplied dataset and checkpoints.
- No standardized cross-model latency benchmark was reported.
- Worker-to-equipment association is not implemented.
- The application is a local demonstration, with no field validation of safety outcomes.

## Future Improvements

- Normalize mixed annotation formats and repeat evaluation.
- Address class imbalance and improve rare-class detection.
- Associate PPE detections with individual workers.
- Calibrate class-specific confidence thresholds on validation data.
- Benchmark inference speed and memory under consistent conditions.
- Extend image inference to video or camera input.

## Source and Licensing

Project details and results are documented in `PPE_Deep_Learning_Business_Case_Study_Report.docx`.

Consult the original dataset metadata, model terms, and dependency licenses before redistributing data or weights. This README does not assign a software or dataset license.

## Contributors

Sharath N S, Pothireddy Avinash Reddy, Mandara D D, Adarsh Shaw, Mayank, and Keshav Prasad.

