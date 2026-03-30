# DeepFake Detector Pro (PySide6)

A production-style desktop deepfake detector built with **PySide6 (Qt for Python)** and TensorFlow inference.

## Highlights

- Premium dark desktop UI (`#0f172a` base)
- Sidebar navigation: Detection, Batch, Settings
- Topbar actions: Open Image, Open Video, Settings
- Drag-and-drop media upload
- Image + video inference (`predict_image`, `predict_video`)
- Threshold/method controls and live logs
- Exportable text reports
- Batch scanning for directories
- Threaded inference (UI stays responsive)

## Architecture

```text
app/
  qt/
    app.py
    main_window.py
    styles.py
    widgets/
      drop_zone.py
      preview_widget.py
      spinner.py
    workers/
      detection_worker.py
      batch_worker.py
  main.py                  # CLI fallback
services/
  inference_service.py
  report_service.py
core/
  types.py
config/
  settings.py
run.py                     # Desktop launcher (default)
```

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch desktop app:

```bash
python run.py
```

Run CLI mode:

```bash
python run.py --cli --image path/to/image.jpg
python run.py --cli --video path/to/video.mp4
```

## Model Notes

`services/inference_service.py` loads a model once from `models/` and adapts preprocessing to model input shape.

Supported files:
- Images: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`
- Videos: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
