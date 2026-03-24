import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import threading
import time
import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QCheckBox,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

# --- 共有フレームバッファ ---
latest_frame = None
latest_preview = None
frame_lock = threading.Lock()

# --- カメラ切り替え用 ---
camera_index = 0
camera_switch_event = threading.Event()
camera_status = ""
status_lock = threading.Lock()

# --- ブラー設定 ---
blur_config = {
    "faces": True,
    "persons": False,
    "screens": False,
    "license_plates": False,
}
blur_config_lock = threading.Lock()


def enumerate_cameras(max_index=5):
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            available.append((i, f"カメラ {i} ({w}x{h})"))
            cap.release()
    time.sleep(0.5)
    return available


def get_model_path(filename='blaze_face_short_range.tflite'):
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, filename)


def blur_region(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size > 0:
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (99, 99), 30)
    return frame


def blur_detection(frame, detection, padding=0.25):
    bb = detection.bounding_box
    px = int(bb.width * padding)
    py = int(bb.height * padding)
    return blur_region(frame,
                       bb.origin_x - px, bb.origin_y - py,
                       bb.origin_x + bb.width + px,
                       bb.origin_y + bb.height + py)


def open_camera(index, retries=5):
    for attempt in range(retries):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            for _ in range(3):
                cap.read()
            return cap
        cap.release()
        print(f"  カメラ {index} を開けません (試行 {attempt + 1}/{retries})")
        time.sleep(0.5)
    return None


def capture_loop():
    global latest_frame, latest_preview, camera_status

    try:
        # 顔検出器
        face_detector = mp_vision.FaceDetector.create_from_options(
            mp_vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=get_model_path('blaze_face_short_range.tflite')
                ),
                min_detection_confidence=0.5,
            )
        )

        # 物体検出器（人物・画面）
        object_detector = mp_vision.ObjectDetector.create_from_options(
            mp_vision.ObjectDetectorOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=get_model_path('efficientdet_lite0.tflite')
                ),
                max_results=10,
                score_threshold=0.4,
                category_allowlist=["person", "tv", "laptop", "cell phone"],
            )
        )

        # ナンバープレート検出器
        plate_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
        )

        cap = open_camera(camera_index)
        if cap is None:
            with status_lock:
                camera_status = "エラー: カメラを開けません"
            return
        info = f"カメラ {camera_index}: {int(cap.get(3))}x{int(cap.get(4))} @ {cap.get(5):.0f}fps"
        print(f"  {info}")
        with status_lock:
            camera_status = info

        while True:
            if camera_switch_event.is_set():
                cap.release()
                cap = open_camera(camera_index)
                if cap is None:
                    with status_lock:
                        camera_status = f"エラー: カメラ {camera_index} を開けません"
                    camera_switch_event.clear()
                    continue
                info = f"カメラ {camera_index}: {int(cap.get(3))}x{int(cap.get(4))} @ {cap.get(5):.0f}fps"
                print(f"  {info}")
                with status_lock:
                    camera_status = info
                camera_switch_event.clear()

            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            output = frame.copy()

            with blur_config_lock:
                config = blur_config.copy()

            # 顔
            if config["faces"]:
                result = face_detector.detect(mp_image)
                for det in result.detections:
                    output = blur_detection(output, det, padding=0.25)

            # 人物・画面（共通の物体検出器）
            if config["persons"] or config["screens"]:
                obj_result = object_detector.detect(mp_image)
                for det in obj_result.detections:
                    label = det.categories[0].category_name
                    if label == "person" and config["persons"]:
                        output = blur_detection(output, det, padding=0.05)
                    elif label in ("tv", "laptop", "cell phone") and config["screens"]:
                        output = blur_detection(output, det, padding=0.02)

            # ナンバープレート
            if config["license_plates"]:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                plates = plate_cascade.detectMultiScale(
                    gray, 1.1, 4, minSize=(60, 20)
                )
                for (x, y, w, h) in plates:
                    output = blur_region(output, x, y, x + w, y + h)

            _, jpeg = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 85])
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            with frame_lock:
                latest_frame = jpeg.tobytes()
                latest_preview = output_rgb

        cap.release()
    except Exception as e:
        import traceback
        print(f"capture_loop エラー: {e}")
        traceback.print_exc()
        with status_lock:
            camera_status = f"エラー: {e}"


# --- MJPEGサーバー ---
class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type',
                         'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                with frame_lock:
                    frame = latest_frame
                if frame:
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
        except (BrokenPipeError, ConnectionResetError):
            pass


# --- PyQt6 GUI ---
PREVIEW_WIDTH = 480


class MainWindow(QWidget):
    def __init__(self, cameras):
        super().__init__()
        self.cameras = cameras
        self.setWindowTitle("Face Blur")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # プレビュー
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #1e1e1e;")
        self.preview_label.setMinimumSize(PREVIEW_WIDTH, 270)
        layout.addWidget(self.preview_label)

        # ブラー対象トグル
        toggle_layout = QHBoxLayout()
        toggle_layout.setContentsMargins(8, 6, 8, 6)

        self.cb_faces = QCheckBox("顔")
        self.cb_faces.setChecked(True)
        self.cb_faces.toggled.connect(lambda v: self._set_config("faces", v))

        self.cb_persons = QCheckBox("人物")
        self.cb_persons.toggled.connect(lambda v: self._set_config("persons", v))

        self.cb_screens = QCheckBox("画面")
        self.cb_screens.toggled.connect(lambda v: self._set_config("screens", v))

        self.cb_plates = QCheckBox("ナンバー")
        self.cb_plates.toggled.connect(lambda v: self._set_config("license_plates", v))

        for cb in (self.cb_faces, self.cb_persons, self.cb_screens, self.cb_plates):
            toggle_layout.addWidget(cb)
        toggle_layout.setSpacing(16)
        toggle_layout.addStretch()

        layout.addLayout(toggle_layout)

        # 下部コントロール
        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(8, 0, 8, 6)

        ctrl.addWidget(QLabel("カメラ:"))
        self.combo = QComboBox()
        for _, label in cameras:
            self.combo.addItem(label)
        self.combo.currentIndexChanged.connect(self.on_camera_change)
        ctrl.addWidget(self.combo)

        ctrl.addStretch()

        self.status_label = QLabel("起動中...")
        self.status_label.setStyleSheet("color: gray; font-size: 12px;")
        ctrl.addWidget(self.status_label)

        layout.addLayout(ctrl)
        self.setLayout(layout)

        # タイマーでプレビュー更新
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_preview)
        self.timer.start(33)

    def _set_config(self, key, value):
        with blur_config_lock:
            blur_config[key] = value

    def on_camera_change(self, idx):
        global camera_index
        if idx < 0:
            return
        new_index = self.cameras[idx][0]
        if new_index == camera_index:
            return
        camera_index = new_index
        camera_switch_event.set()
        self.status_label.setText(f"カメラ {new_index} に切り替え中...")

    def update_preview(self):
        with frame_lock:
            preview = latest_preview

        if preview is not None:
            h, w = preview.shape[:2]
            scale = PREVIEW_WIDTH / w
            new_w = PREVIEW_WIDTH
            new_h = int(h * scale)
            resized = cv2.resize(preview, (new_w, new_h))
            qimg = QImage(resized.data, new_w, new_h,
                          new_w * 3, QImage.Format.Format_RGB888)
            self.preview_label.setPixmap(QPixmap.fromImage(qimg))

        with status_lock:
            s = camera_status
        if s:
            self.status_label.setText(s)


if __name__ == '__main__':
    print("カメラを検出中...")
    cameras = enumerate_cameras()
    if not cameras:
        print("エラー: カメラが見つかりません")
        sys.exit(1)
    print(f"{len(cameras)} 台のカメラを検出:")
    for _, label in cameras:
        print(f"  {label}")

    # キャプチャスレッド
    t_capture = threading.Thread(target=capture_loop, daemon=True)
    t_capture.start()

    # MJPEGサーバースレッド
    port = 8080
    server = HTTPServer(('localhost', port), MJPEGHandler)
    t_server = threading.Thread(target=server.serve_forever, daemon=True)
    t_server.start()
    print(f"ストリーム配信中: http://localhost:{port}/")

    # PyQt6 GUI（メインスレッド）
    app = QApplication(sys.argv)
    window = MainWindow(cameras)
    window.show()
    sys.exit(app.exec())
