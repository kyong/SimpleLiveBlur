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
    QLabel, QComboBox,
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


def blur_faces(frame, detections, padding=0.25):
    h, w = frame.shape[:2]
    result = frame.copy()
    if not detections:
        return result
    for det in detections:
        bb = det.bounding_box
        px = int(bb.width * padding)
        py = int(bb.height * padding)
        x1 = max(0, bb.origin_x - px)
        y1 = max(0, bb.origin_y - py)
        x2 = min(w, bb.origin_x + bb.width + px)
        y2 = min(h, bb.origin_y + bb.height + py)
        roi = result[y1:y2, x1:x2]
        if roi.size > 0:
            result[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (99, 99), 30)
    return result


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


def get_model_path():
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, 'blaze_face_short_range.tflite')


def capture_loop():
    global latest_frame, latest_preview, camera_status

    try:
        base_options = mp_python.BaseOptions(
            model_asset_path=get_model_path()
        )
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.5,
        )
        detector = mp_vision.FaceDetector.create_from_options(options)

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
            result = detector.detect(mp_image)
            output = blur_faces(frame, result.detections)
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

        # 下部コントロール
        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(8, 6, 8, 6)

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
