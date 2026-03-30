"""Main PySide6 desktop window for DeepFake Detector."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QPropertyAnimation, Qt, QThread
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from config.settings import MODELS_DIR, SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS
from core.types import BatchItemResult, DetectionResult
from services.inference_service import InferenceService
from services.report_service import export_detection_report

from .styles import build_stylesheet
from .widgets.drop_zone import DropZoneWidget
from .widgets.preview_widget import PreviewWidget
from .widgets.spinner import SpinnerWidget
from .workers.batch_worker import BatchWorker
from .workers.detection_worker import DetectionWorker


class MainWindow(QMainWindow):
    """Primary desktop shell and interaction coordinator."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DeepFake Detector Pro")
        self.resize(1420, 920)
        self.setMinimumSize(1140, 760)

        self.setStyleSheet(build_stylesheet())

        self.inference_service = InferenceService(model_dir=MODELS_DIR)

        self.current_media_path: str | None = None
        self.current_media_type: str | None = None
        self.current_result: DetectionResult | None = None

        self._thread: QThread | None = None
        self._batch_thread: QThread | None = None
        self._detection_worker: DetectionWorker | None = None
        self._batch_worker: BatchWorker | None = None

        self._animations: list[QPropertyAnimation] = []

        self._build_ui()
        self._register_shortcuts()
        self.append_log(f"Application ready | runtime: {self.inference_service.runtime_description}")

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(14)

        sidebar = self._build_sidebar()
        content_shell = self._build_content_shell()

        root_layout.addWidget(sidebar)
        root_layout.addWidget(content_shell, stretch=1)

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(220)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 18, 16, 18)
        layout.setSpacing(12)

        title = QLabel("DeepFake Pro")
        title.setObjectName("BrandTitle")

        subtitle = QLabel("Desktop Intelligence")
        subtitle.setObjectName("MutedLabel")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.nav_buttons: dict[str, QPushButton] = {}
        for key, label in (
            ("detection", "Detection"),
            ("batch", "Batch"),
            ("settings", "Settings"),
        ):
            button = QPushButton(label)
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.clicked.connect(lambda _, target=key: self.switch_page(target))
            self.nav_buttons[key] = button
            layout.addWidget(button)

        layout.addStretch(1)

        self.sidebar_runtime_label = QLabel(
            f"Runtime: {self.inference_service.runtime_description}"
        )
        self.sidebar_runtime_label.setObjectName("MutedLabel")
        self.sidebar_runtime_label.setWordWrap(True)
        layout.addWidget(self.sidebar_runtime_label)

        return sidebar

    def _build_content_shell(self) -> QWidget:
        shell = QWidget()
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(12)

        self.topbar = self._build_topbar()
        shell_layout.addWidget(self.topbar)

        self.stack = QStackedWidget()
        shell_layout.addWidget(self.stack, stretch=1)

        self.detection_page = self._build_detection_page()
        self.batch_page = self._build_batch_page()
        self.settings_page = self._build_settings_page()

        self.stack.addWidget(self.detection_page)
        self.stack.addWidget(self.batch_page)
        self.stack.addWidget(self.settings_page)

        self.switch_page("detection")
        return shell

    def _build_topbar(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("TopBar")
        self._apply_shadow(frame)

        layout = QHBoxLayout(frame)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(8)

        self.btn_open_image = QPushButton("Open Image")
        self.btn_open_image.setObjectName("PrimaryButton")
        self.btn_open_image.clicked.connect(self.open_image)

        self.btn_open_video = QPushButton("Open Video")
        self.btn_open_video.setObjectName("PrimaryButton")
        self.btn_open_video.clicked.connect(self.open_video)

        self.btn_settings = QPushButton("Settings")
        self.btn_settings.setObjectName("SecondaryButton")
        self.btn_settings.clicked.connect(lambda: self.switch_page("settings"))

        layout.addWidget(self.btn_open_image)
        layout.addWidget(self.btn_open_video)
        layout.addStretch(1)
        layout.addWidget(self.btn_settings)

        return frame

    def _build_detection_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.drop_zone = DropZoneWidget()
        self.drop_zone.clicked.connect(self._browse_from_drop_zone)
        self.drop_zone.file_dropped.connect(self.set_media_path)
        layout.addWidget(self.drop_zone)

        body = QHBoxLayout()
        body.setSpacing(12)
        layout.addLayout(body, stretch=1)

        left_column = QVBoxLayout()
        left_column.setSpacing(12)
        body.addLayout(left_column, stretch=5)

        self.preview_card = QFrame()
        self.preview_card.setObjectName("Card")
        self._apply_shadow(self.preview_card)
        preview_card_layout = QVBoxLayout(self.preview_card)
        preview_card_layout.setContentsMargins(14, 14, 14, 14)
        preview_card_layout.setSpacing(8)

        preview_title = QLabel("Preview")
        preview_title.setObjectName("MutedLabel")

        self.preview_widget = PreviewWidget()

        preview_card_layout.addWidget(preview_title)
        preview_card_layout.addWidget(self.preview_widget, stretch=1)
        left_column.addWidget(self.preview_card, stretch=1)

        self.controls_card = QFrame()
        self.controls_card.setObjectName("Card")
        self._apply_shadow(self.controls_card)
        controls_layout = QVBoxLayout(self.controls_card)
        controls_layout.setContentsMargins(14, 14, 14, 14)
        controls_layout.setSpacing(10)

        controls_title = QLabel("Detection Controls")
        controls_title.setObjectName("MutedLabel")

        threshold_row = QHBoxLayout()
        threshold_label = QLabel("Threshold")
        threshold_label.setObjectName("MutedLabel")
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(30, 95)
        self.threshold_slider.setValue(50)
        self.threshold_slider.valueChanged.connect(self._sync_threshold_label)
        self.threshold_value_label = QLabel("0.50")
        self.threshold_value_label.setMinimumWidth(40)

        threshold_row.addWidget(threshold_label)
        threshold_row.addWidget(self.threshold_slider, stretch=1)
        threshold_row.addWidget(self.threshold_value_label)

        method_row = QHBoxLayout()
        method_label = QLabel("Method")
        method_label.setObjectName("MutedLabel")
        self.method_combo = QComboBox()
        self.method_combo.addItem("Balanced", "balanced")
        self.method_combo.addItem("Fast", "fast")
        self.method_combo.addItem("Robust", "robust")
        self.method_combo.addItem("Face Focus", "face-focus")
        self.method_combo.addItem("Frequency", "frequency")
        method_row.addWidget(method_label)
        method_row.addWidget(self.method_combo, stretch=1)

        action_row = QHBoxLayout()
        self.run_button = QPushButton("Run Scan")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.clicked.connect(self.run_detection)

        self.spinner = SpinnerWidget()
        self.spinner_label = QLabel("Scanning...")
        self.spinner_label.setObjectName("MutedLabel")
        self.spinner_label.hide()

        self.busy_bar = QLabel()
        self.busy_bar.setVisible(False)

        action_row.addWidget(self.run_button)
        action_row.addStretch(1)
        action_row.addWidget(self.spinner_label)
        action_row.addWidget(self.spinner)

        controls_layout.addWidget(controls_title)
        controls_layout.addLayout(threshold_row)
        controls_layout.addLayout(method_row)
        controls_layout.addLayout(action_row)
        left_column.addWidget(self.controls_card)

        right_column = QVBoxLayout()
        right_column.setSpacing(12)
        body.addLayout(right_column, stretch=4)

        self.results_card = QFrame()
        self.results_card.setObjectName("Card")
        self._apply_shadow(self.results_card)
        results_layout = QVBoxLayout(self.results_card)
        results_layout.setContentsMargins(14, 14, 14, 14)
        results_layout.setSpacing(10)

        results_title = QLabel("Results")
        results_title.setObjectName("MutedLabel")

        badge_row = QHBoxLayout()
        self.result_badge = QLabel("NO RESULT")
        self.result_badge.setObjectName("ResultBadge")

        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setObjectName("MutedLabel")

        badge_row.addWidget(self.result_badge)
        badge_row.addStretch(1)
        badge_row.addWidget(self.confidence_label)

        self.logs_panel = QTextEdit()
        self.logs_panel.setReadOnly(True)
        self.logs_panel.setMinimumHeight(240)

        export_row = QHBoxLayout()
        self.export_button = QPushButton("Export Report")
        self.export_button.setObjectName("SecondaryButton")
        self.export_button.clicked.connect(self.export_current_report)
        self.export_button.setEnabled(False)

        export_row.addStretch(1)
        export_row.addWidget(self.export_button)

        results_layout.addWidget(results_title)
        results_layout.addLayout(badge_row)
        results_layout.addWidget(self.logs_panel, stretch=1)
        results_layout.addLayout(export_row)

        right_column.addWidget(self.results_card, stretch=1)

        self._sync_threshold_label()
        return page

    def _build_batch_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        actions_card = QFrame()
        actions_card.setObjectName("Card")
        self._apply_shadow(actions_card)
        actions_layout = QHBoxLayout(actions_card)
        actions_layout.setContentsMargins(14, 12, 14, 12)
        actions_layout.setSpacing(8)

        self.batch_folder_label = QLabel("No folder selected")
        self.batch_folder_label.setObjectName("MutedLabel")

        self.select_batch_btn = QPushButton("Select Folder")
        self.select_batch_btn.setObjectName("SecondaryButton")
        self.select_batch_btn.clicked.connect(self.select_batch_folder)

        self.run_batch_btn = QPushButton("Run Batch Scan")
        self.run_batch_btn.setObjectName("PrimaryButton")
        self.run_batch_btn.clicked.connect(self.run_batch_scan)

        actions_layout.addWidget(self.select_batch_btn)
        actions_layout.addWidget(self.run_batch_btn)
        actions_layout.addStretch(1)
        actions_layout.addWidget(self.batch_folder_label)

        table_card = QFrame()
        table_card.setObjectName("Card")
        self._apply_shadow(table_card)
        table_layout = QVBoxLayout(table_card)
        table_layout.setContentsMargins(14, 14, 14, 14)
        table_layout.setSpacing(8)

        table_title = QLabel("Batch Results")
        table_title.setObjectName("MutedLabel")

        self.batch_table = QTableWidget(0, 4)
        self.batch_table.setHorizontalHeaderLabels(["File", "Status", "p(fake)", "Details"])
        self.batch_table.horizontalHeader().setStretchLastSection(True)
        self.batch_table.verticalHeader().setVisible(False)
        self.batch_table.setAlternatingRowColors(False)

        table_layout.addWidget(table_title)
        table_layout.addWidget(self.batch_table, stretch=1)

        layout.addWidget(actions_card)
        layout.addWidget(table_card, stretch=1)

        return page

    def _build_settings_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        card = QFrame()
        card.setObjectName("Card")
        self._apply_shadow(card)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 16, 16, 16)
        card_layout.setSpacing(10)

        title = QLabel("Settings")
        title.setObjectName("BrandTitle")

        model_path_label = QLabel(f"Model directory: {MODELS_DIR}")
        model_path_label.setObjectName("MutedLabel")
        model_path_label.setWordWrap(True)

        runtime_label = QLabel(f"Inference runtime: {self.inference_service.runtime_description}")
        runtime_label.setObjectName("MutedLabel")
        runtime_label.setWordWrap(True)

        hint = QLabel(
            "Tip: You can switch threshold and method from the Detection page. "
            "Inference engine keeps models in memory for faster repeated scans."
        )
        hint.setObjectName("MutedLabel")
        hint.setWordWrap(True)

        card_layout.addWidget(title)
        card_layout.addWidget(model_path_label)
        card_layout.addWidget(runtime_label)
        card_layout.addWidget(hint)
        card_layout.addStretch(1)

        layout.addWidget(card)
        layout.addStretch(1)

        return page

    def _register_shortcuts(self) -> None:
        open_image_action = QAction(self)
        open_image_action.setShortcut("Ctrl+O")
        open_image_action.triggered.connect(self.open_image)
        self.addAction(open_image_action)

    def _apply_shadow(self, widget: QWidget) -> None:
        shadow = QGraphicsDropShadowEffect(widget)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 10)
        shadow.setColor(Qt.GlobalColor.black)
        widget.setGraphicsEffect(shadow)

    def _sync_threshold_label(self) -> None:
        self.threshold_value_label.setText(f"{self.threshold_slider.value() / 100:.2f}")

    def current_threshold(self) -> float:
        return self.threshold_slider.value() / 100.0

    def current_method(self) -> str:
        return str(self.method_combo.currentData())

    def switch_page(self, target: str) -> None:
        index_map = {"detection": 0, "batch": 1, "settings": 2}
        if target not in index_map:
            return

        self.stack.setCurrentIndex(index_map[target])
        for key, button in self.nav_buttons.items():
            button.setChecked(key == target)

    def _browse_from_drop_zone(self) -> None:
        selected_filter = "Media Files (*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Media", str(Path.home()), selected_filter)
        if file_path:
            self.set_media_path(file_path)

    def open_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            str(Path.home()),
            "Image Files (*.jpg *.jpeg *.png *.bmp *.webp *.tiff)",
        )
        if file_path:
            self.set_media_path(file_path)

    def open_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            str(Path.home()),
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)",
        )
        if file_path:
            self.set_media_path(file_path)

    def set_media_path(self, path: str) -> None:
        suffix = Path(path).suffix.lower()
        if suffix in SUPPORTED_IMAGE_EXTENSIONS:
            media_type = "image"
            self.preview_widget.set_image(path)
        elif suffix in SUPPORTED_VIDEO_EXTENSIONS:
            media_type = "video"
            self.preview_widget.set_video(path)
        else:
            QMessageBox.warning(self, "Unsupported File", "Please select a supported image or video file.")
            return

        self.current_media_path = path
        self.current_media_type = media_type
        self.append_log(f"Loaded {media_type}: {Path(path).name}")
        self._fade_in(self.preview_card)

    def _set_running_state(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        if running:
            self.spinner.start()
            self.spinner_label.show()
        else:
            self.spinner.stop()
            self.spinner_label.hide()

    def run_detection(self) -> None:
        if not self.current_media_path or not self.current_media_type:
            QMessageBox.information(self, "No Media", "Open or drop an image/video before scanning.")
            return

        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "Scan Running", "A scan is already running. Please wait for it to finish.")
            return

        self._set_running_state(True)
        self.append_log(
            "Scan requested | "
            f"file={Path(self.current_media_path).name} | "
            f"type={self.current_media_type} | "
            f"method={self.current_method()} | "
            f"threshold={self.current_threshold():.2f}"
        )

        worker = DetectionWorker(
            inference_service=self.inference_service,
            media_path=self.current_media_path,
            media_type=self.current_media_type,
            threshold=self.current_threshold(),
            method=self.current_method(),
        )

        thread = QThread(self)
        worker.moveToThread(thread)

        worker.log.connect(self.append_log)
        worker.finished.connect(self._on_detection_finished)
        worker.error.connect(self._on_detection_error)

        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_detection_thread_finished)

        thread.started.connect(lambda: self.append_log("Scan worker thread started"))
        thread.started.connect(worker.run)

        self._detection_worker = worker
        self._thread = thread
        thread.start()

    def _on_detection_finished(self, result: DetectionResult) -> None:
        self.current_result = result
        self.export_button.setEnabled(True)
        self._set_running_state(False)

        status_text = "FAKE" if result.is_fake else "REAL"
        if result.is_fake:
            badge_style = "background-color: #7f1d1d; color: #fecaca;"
        else:
            badge_style = "background-color: #14532d; color: #bbf7d0;"

        self.result_badge.setText(status_text)
        self.result_badge.setStyleSheet(
            "padding: 6px 12px; border-radius: 16px; font-size: 13px; font-weight: 700; " + badge_style
        )
        self.confidence_label.setText(
            f"Confidence: {result.confidence * 100:.2f}% | p(fake): {result.fake_probability * 100:.2f}%"
        )

        self.append_log(
            f"Result {status_text} | confidence={result.confidence * 100:.2f}% | "
            f"frames={result.frames_analyzed} | runtime={result.metadata.get('runtime_provider', 'unknown')}"
        )

        self._fade_in(self.results_card)

    def _on_detection_error(self, message: str) -> None:
        self._set_running_state(False)
        self.append_log(f"ERROR: {message}")
        QMessageBox.critical(self, "Detection Error", message)

    def _on_detection_thread_finished(self) -> None:
        self._thread = None
        self._detection_worker = None
        if self.spinner.isVisible():
            self._set_running_state(False)
            self.append_log("Scan stopped before completion. Please run it again.")

    def export_current_report(self) -> None:
        if self.current_result is None:
            QMessageBox.information(self, "No Result", "Run a detection first before exporting.")
            return

        default_name = f"{Path(self.current_result.media_path).stem}_report.txt"
        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Export Report",
            str(Path.home() / default_name),
            "Text File (*.txt)",
        )
        if not destination:
            return

        output_path = export_detection_report(self.current_result, destination)
        self.append_log(f"Report exported: {output_path}")
        QMessageBox.information(self, "Report Exported", f"Saved report to:\n{output_path}")

    def select_batch_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Batch Folder", str(Path.home()))
        if not folder:
            return
        self.batch_folder_label.setText(folder)

    def run_batch_scan(self) -> None:
        folder = self.batch_folder_label.text().strip()
        if not folder or folder == "No folder selected":
            QMessageBox.information(self, "No Folder", "Select a folder before starting batch scan.")
            return

        if self._batch_thread is not None and self._batch_thread.isRunning():
            QMessageBox.information(self, "Batch Running", "A batch scan is already running.")
            return

        self.batch_table.setRowCount(0)
        self.run_batch_btn.setEnabled(False)
        self.select_batch_btn.setEnabled(False)

        worker = BatchWorker(
            inference_service=self.inference_service,
            directory_path=folder,
            threshold=self.current_threshold(),
            method=self.current_method(),
        )

        thread = QThread(self)
        worker.moveToThread(thread)

        worker.log.connect(self.append_log)
        worker.item_complete.connect(self._append_batch_result)
        worker.finished.connect(self._on_batch_finished)
        worker.error.connect(self._on_batch_error)

        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_batch_thread_finished)

        thread.started.connect(worker.run)

        self._batch_worker = worker
        self._batch_thread = thread
        thread.start()

    def _append_batch_result(self, item: BatchItemResult) -> None:
        row = self.batch_table.rowCount()
        self.batch_table.insertRow(row)

        file_name = Path(item.media_path).name
        self.batch_table.setItem(row, 0, QTableWidgetItem(file_name))
        self.batch_table.setItem(row, 1, QTableWidgetItem(item.status))

        prob_text = "--" if item.fake_probability is None else f"{item.fake_probability:.3f}"
        self.batch_table.setItem(row, 2, QTableWidgetItem(prob_text))

        details = item.error or (f"Confidence: {item.confidence:.3f}" if item.confidence is not None else "")
        self.batch_table.setItem(row, 3, QTableWidgetItem(details))

    def _on_batch_finished(self, results: list[BatchItemResult]) -> None:
        self.run_batch_btn.setEnabled(True)
        self.select_batch_btn.setEnabled(True)

        fake_count = sum(1 for item in results if item.status == "FAKE")
        real_count = sum(1 for item in results if item.status == "REAL")
        err_count = sum(1 for item in results if item.status == "ERROR")

        self.append_log(f"Batch summary | fake={fake_count} real={real_count} errors={err_count}")
        self._fade_in(self.batch_table)

    def _on_batch_error(self, message: str) -> None:
        self.run_batch_btn.setEnabled(True)
        self.select_batch_btn.setEnabled(True)
        self.append_log(f"BATCH ERROR: {message}")
        QMessageBox.critical(self, "Batch Error", message)

    def _on_batch_thread_finished(self) -> None:
        self._batch_thread = None
        self._batch_worker = None
        if not self.run_batch_btn.isEnabled():
            self.run_batch_btn.setEnabled(True)
            self.select_batch_btn.setEnabled(True)
            self.append_log("Batch scan stopped before completion. Please run it again.")

    def append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs_panel.append(f"[{timestamp}] {message}")

    def _fade_in(self, widget: QWidget, duration_ms: int = 260) -> None:
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)

        animation = QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(duration_ms)
        animation.setStartValue(0.35)
        animation.setEndValue(1.0)
        animation.start()

        # Keep reference to avoid immediate GC.
        self._animations.append(animation)
        if len(self._animations) > 12:
            self._animations.pop(0)
