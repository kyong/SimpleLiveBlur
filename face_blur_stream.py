import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import threading
import asyncio
import json
import time
import sys
import os
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QCheckBox, QSlider, QGroupBox,
    QFormLayout,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

# --- 共有フレームバッファ ---
latest_frame = None
latest_preview = None
frame_lock = threading.Lock()

# --- カメラ切り替え用 ---
camera_index = 0
camera_switch_event = threading.Event()
stop_event = threading.Event()
camera_status = ""
status_lock = threading.Lock()

# --- ブラー設定 ---
blur_config = {
    "faces": True,
    "persons": False,
    "screens": False,
    "license_plates": False,
    "blur_strength": 99,        # カーネルサイズ（奇数、1〜199）
    "blur_sigma": 30,           # 標準偏差（1〜100）
    "face_threshold": 0.5,      # 顔検出の閾値（0.0〜1.0）
    "object_threshold": 0.4,    # 物体検出の閾値（0.0〜1.0）
    "history_frames": 5,        # 検出マージフレーム数（0=OFF）
}
blur_config_lock = threading.Lock()


def _video_capture(index):
    if sys.platform == 'darwin':
        return cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(index)


def enumerate_cameras(max_index=5):
    available = []
    for i in range(max_index):
        cap = _video_capture(i)
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


def blur_region(frame, x1, y1, x2, y2, ksize=99, sigma=30):
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size > 0:
        k = ksize if ksize % 2 == 1 else ksize + 1
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), sigma)
    return frame


def blur_detection(frame, detection, padding=0.25, ksize=99, sigma=30):
    bb = detection.bounding_box
    px = int(bb.width * padding)
    py = int(bb.height * padding)
    return blur_region(frame,
                       bb.origin_x - px, bb.origin_y - py,
                       bb.origin_x + bb.width + px,
                       bb.origin_y + bb.height + py,
                       ksize=ksize, sigma=sigma)


def open_camera(index, retries=5):
    for attempt in range(retries):
        cap = _video_capture(index)
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

        # 検出矩形の履歴（直近フレーム分を保持して検出漏れを補完）
        rect_history = deque(maxlen=5)

        while not stop_event.is_set():
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
                rect_history.clear()

            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            output = frame.copy()

            with blur_config_lock:
                config = blur_config.copy()

            ksize = config["blur_strength"]
            sigma = config["blur_sigma"]
            face_th = config["face_threshold"]
            obj_th = config["object_threshold"]

            # 現フレームの検出矩形を収集
            current_rects = []

            # 顔
            if config["faces"]:
                result = face_detector.detect(mp_image)
                for det in result.detections:
                    if det.categories[0].score >= face_th:
                        bb = det.bounding_box
                        pad = 0.25
                        px, py = int(bb.width * pad), int(bb.height * pad)
                        current_rects.append((
                            bb.origin_x - px, bb.origin_y - py,
                            bb.origin_x + bb.width + px,
                            bb.origin_y + bb.height + py,
                        ))

            # 人物・画面（共通の物体検出器）
            if config["persons"] or config["screens"]:
                obj_result = object_detector.detect(mp_image)
                for det in obj_result.detections:
                    cat = det.categories[0]
                    if cat.score < obj_th:
                        continue
                    bb = det.bounding_box
                    if cat.category_name == "person" and config["persons"]:
                        pad = 0.05
                        px, py = int(bb.width * pad), int(bb.height * pad)
                        current_rects.append((
                            bb.origin_x - px, bb.origin_y - py,
                            bb.origin_x + bb.width + px,
                            bb.origin_y + bb.height + py,
                        ))
                    elif cat.category_name in ("tv", "laptop", "cell phone") and config["screens"]:
                        pad = 0.02
                        px, py = int(bb.width * pad), int(bb.height * pad)
                        current_rects.append((
                            bb.origin_x - px, bb.origin_y - py,
                            bb.origin_x + bb.width + px,
                            bb.origin_y + bb.height + py,
                        ))

            # ナンバープレート
            if config["license_plates"]:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                plates = plate_cascade.detectMultiScale(
                    gray, 1.1, 4, minSize=(60, 20)
                )
                for (x, y, w, h) in plates:
                    current_rects.append((x, y, x + w, y + h))

            # 履歴フレーム数を設定から反映
            history_frames = config["history_frames"]
            if history_frames != rect_history.maxlen:
                rect_history = deque(rect_history, maxlen=max(1, history_frames))

            # 履歴に追加
            rect_history.append(current_rects)

            # 現フレーム＋過去フレームの全矩形をマージしてブラー
            if history_frames > 0:
                for rects in rect_history:
                    for (x1, y1, x2, y2) in rects:
                        output = blur_region(output, x1, y1, x2, y2,
                                             ksize=ksize, sigma=sigma)
            else:
                for (x1, y1, x2, y2) in current_rects:
                    output = blur_region(output, x1, y1, x2, y2,
                                         ksize=ksize, sigma=sigma)

            _, jpeg = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 85])
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            with frame_lock:
                latest_frame = jpeg.tobytes()
                latest_preview = output_rgb

        cap.release()
        print("カメラを解放しました")
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


# --- WebSocket制御サーバー ---
# 設定キーの型・範囲定義
_CONFIG_SCHEMA = {
    "faces": bool,
    "persons": bool,
    "screens": bool,
    "license_plates": bool,
    "blur_strength": (int, 1, 199),
    "blur_sigma": (int, 1, 100),
    "face_threshold": (float, 0.01, 1.0),
    "object_threshold": (float, 0.01, 1.0),
    "history_frames": (int, 0, 15),
}

# GUIとWebSocket間の同期用シグナル
ws_config_update = None  # MainWindow設定後にセットされる


def _validate_config_value(key, value):
    """設定値をバリデーションし、正規化した値を返す。不正な場合はNone。"""
    schema = _CONFIG_SCHEMA.get(key)
    if schema is None:
        return None
    if schema is bool:
        if isinstance(value, bool):
            return value
        return None
    typ, lo, hi = schema
    try:
        v = typ(value)
    except (ValueError, TypeError):
        return None
    return max(lo, min(hi, v))


def _get_config_snapshot():
    with blur_config_lock:
        return blur_config.copy()


async def _ws_handler(websocket):
    """
    WebSocketメッセージ形式（JSON）:
      {"action": "get_config"}
      {"action": "set_config", "key": "faces", "value": true}
      {"action": "set_config", "config": {"faces": true, "blur_strength": 50}}
      {"action": "get_status"}
    """
    async for raw in websocket:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.send(json.dumps(
                {"error": "invalid JSON"}, ensure_ascii=False))
            continue

        action = msg.get("action")

        if action == "get_config":
            await websocket.send(json.dumps(
                {"config": _get_config_snapshot()}, ensure_ascii=False))

        elif action == "set_config":
            # 単一キー
            if "key" in msg and "value" in msg:
                pairs = {msg["key"]: msg["value"]}
            # 一括
            elif "config" in msg and isinstance(msg["config"], dict):
                pairs = msg["config"]
            else:
                await websocket.send(json.dumps(
                    {"error": "set_config requires 'key'+'value' or 'config'"},
                    ensure_ascii=False))
                continue

            applied = {}
            errors = {}
            for k, v in pairs.items():
                validated = _validate_config_value(k, v)
                if validated is None:
                    errors[k] = f"invalid value: {v!r}"
                else:
                    with blur_config_lock:
                        blur_config[k] = validated
                    applied[k] = validated

            # GUIに同期通知
            if applied and ws_config_update is not None:
                ws_config_update.emit()

            resp = {"applied": applied}
            if errors:
                resp["errors"] = errors
            await websocket.send(json.dumps(resp, ensure_ascii=False))

        elif action == "get_status":
            with status_lock:
                s = camera_status
            await websocket.send(json.dumps(
                {"status": s, "running": not stop_event.is_set()},
                ensure_ascii=False))

        else:
            await websocket.send(json.dumps(
                {"error": f"unknown action: {action}"}, ensure_ascii=False))


def start_ws_server(port):
    """WebSocketサーバーを別スレッドで起動。ポート競合時は自動で空きポートへ。"""
    async def _run(p):
        async with websockets.serve(_ws_handler, "localhost", p):
            await asyncio.Future()  # run forever

    actual_port = port
    loop = asyncio.new_event_loop()

    for p in [port] + list(range(port + 1, port + 20)):
        try:
            # ポートが使えるかテスト
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("localhost", p))
            s.close()
            actual_port = p
            break
        except OSError:
            continue
    else:
        print(f"警告: WebSocketサーバー用の空きポートが見つかりません ({port}-{port+19})")
        return None

    def _thread_target():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_run(actual_port))

    t = threading.Thread(target=_thread_target, daemon=True)
    t.start()
    print(f"WebSocket制御: ws://localhost:{actual_port}/")
    return actual_port


# --- PyQt6 GUI ---
PREVIEW_WIDTH = 480


class MainWindow(QWidget):
    config_changed_externally = pyqtSignal()

    def __init__(self, cameras):
        super().__init__()
        self.cameras = cameras
        self.setWindowTitle("SimpleLiveBlur")
        self.config_changed_externally.connect(self._sync_gui_from_config)

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

        # 詳細設定（折りたたみ）
        self.detail_toggle = QCheckBox("詳細設定 ▶")
        self.detail_toggle.setStyleSheet("padding: 4px 8px;")
        self.detail_toggle.toggled.connect(self._toggle_details)
        layout.addWidget(self.detail_toggle)

        self.detail_widget = QWidget()
        sliders = QFormLayout()
        sliders.setContentsMargins(8, 0, 8, 4)
        sliders.setHorizontalSpacing(12)

        self.sl_strength = self._make_slider(1, 199, 99, 2)
        self.sl_strength_label = QLabel("99")
        self.sl_strength.valueChanged.connect(self._on_strength_changed)
        strength_row = QHBoxLayout()
        strength_row.addWidget(self.sl_strength)
        strength_row.addWidget(self.sl_strength_label)
        sliders.addRow("ブラー強度:", strength_row)

        self.sl_sigma = self._make_slider(1, 100, 30, 1)
        self.sl_sigma_label = QLabel("30")
        self.sl_sigma.valueChanged.connect(self._on_sigma_changed)
        sigma_row = QHBoxLayout()
        sigma_row.addWidget(self.sl_sigma)
        sigma_row.addWidget(self.sl_sigma_label)
        sliders.addRow("ブラー拡散:", sigma_row)

        self.sl_face_th = self._make_slider(1, 100, 50, 1)
        self.sl_face_th_label = QLabel("0.50")
        self.sl_face_th.valueChanged.connect(self._on_face_th_changed)
        face_th_row = QHBoxLayout()
        face_th_row.addWidget(self.sl_face_th)
        face_th_row.addWidget(self.sl_face_th_label)
        sliders.addRow("顔検出閾値:", face_th_row)

        self.sl_obj_th = self._make_slider(1, 100, 40, 1)
        self.sl_obj_th_label = QLabel("0.40")
        self.sl_obj_th.valueChanged.connect(self._on_obj_th_changed)
        obj_th_row = QHBoxLayout()
        obj_th_row.addWidget(self.sl_obj_th)
        obj_th_row.addWidget(self.sl_obj_th_label)
        sliders.addRow("物体検出閾値:", obj_th_row)

        self.sl_history = self._make_slider(0, 15, 5, 1)
        self.sl_history_label = QLabel("5")
        self.sl_history.valueChanged.connect(self._on_history_changed)
        history_row = QHBoxLayout()
        history_row.addWidget(self.sl_history)
        history_row.addWidget(self.sl_history_label)
        sliders.addRow("検出マージ:", history_row)

        self.detail_widget.setLayout(sliders)
        self.detail_widget.setVisible(False)
        layout.addWidget(self.detail_widget)

        # OBS接続ヘルプ
        self.help_widget = QLabel()
        self.help_widget.setTextFormat(Qt.TextFormat.RichText)
        self.help_widget.setWordWrap(True)
        self.help_widget.setStyleSheet(
            "background-color: #2a2a3a; color: #ccc; font-size: 11px;"
            "padding: 6px 10px; border-radius: 4px; margin: 4px 8px;"
        )
        self.help_widget.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.help_widget)

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

    @staticmethod
    def _make_slider(min_val, max_val, default, step):
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(min_val, max_val)
        sl.setValue(default)
        sl.setSingleStep(step)
        sl.setFixedWidth(200)
        return sl

    def set_port(self, port, ws_port=None):
        lines = [f'📡 配信URL: <b>http://localhost:{port}/</b>']
        if ws_port is not None:
            lines.append(f'🔌 WebSocket: <b>ws://localhost:{ws_port}/</b>')
        lines.append(
            '　OBS → ソース追加 → <b>メディアソース</b> → '
            '「ローカルファイル」OFF → 配信URLを入力'
        )
        self.help_widget.setText('<br>'.join(lines))

    def _toggle_details(self, checked):
        self.detail_widget.setVisible(checked)
        self.detail_toggle.setText("詳細設定 ▼" if checked else "詳細設定 ▶")
        self.adjustSize()

    def _set_config(self, key, value):
        with blur_config_lock:
            blur_config[key] = value

    def _on_strength_changed(self, v):
        v = v if v % 2 == 1 else v + 1
        self.sl_strength_label.setText(str(v))
        self._set_config("blur_strength", v)

    def _on_sigma_changed(self, v):
        self.sl_sigma_label.setText(str(v))
        self._set_config("blur_sigma", v)

    def _on_face_th_changed(self, v):
        val = v / 100.0
        self.sl_face_th_label.setText(f"{val:.2f}")
        self._set_config("face_threshold", val)

    def _on_obj_th_changed(self, v):
        val = v / 100.0
        self.sl_obj_th_label.setText(f"{val:.2f}")
        self._set_config("object_threshold", val)

    def _on_history_changed(self, v):
        self.sl_history_label.setText(str(v) if v > 0 else "OFF")
        self._set_config("history_frames", v)

    def _sync_gui_from_config(self):
        """WebSocketから設定が変更されたとき、GUIを同期する。"""
        with blur_config_lock:
            config = blur_config.copy()
        # blockSignals で再帰的な _set_config 呼び出しを防ぐ
        self.cb_faces.blockSignals(True)
        self.cb_faces.setChecked(config["faces"])
        self.cb_faces.blockSignals(False)

        self.cb_persons.blockSignals(True)
        self.cb_persons.setChecked(config["persons"])
        self.cb_persons.blockSignals(False)

        self.cb_screens.blockSignals(True)
        self.cb_screens.setChecked(config["screens"])
        self.cb_screens.blockSignals(False)

        self.cb_plates.blockSignals(True)
        self.cb_plates.setChecked(config["license_plates"])
        self.cb_plates.blockSignals(False)

        self.sl_strength.blockSignals(True)
        self.sl_strength.setValue(config["blur_strength"])
        self.sl_strength_label.setText(str(config["blur_strength"]))
        self.sl_strength.blockSignals(False)

        self.sl_sigma.blockSignals(True)
        self.sl_sigma.setValue(config["blur_sigma"])
        self.sl_sigma_label.setText(str(config["blur_sigma"]))
        self.sl_sigma.blockSignals(False)

        self.sl_face_th.blockSignals(True)
        self.sl_face_th.setValue(int(config["face_threshold"] * 100))
        self.sl_face_th_label.setText(f"{config['face_threshold']:.2f}")
        self.sl_face_th.blockSignals(False)

        self.sl_obj_th.blockSignals(True)
        self.sl_obj_th.setValue(int(config["object_threshold"] * 100))
        self.sl_obj_th_label.setText(f"{config['object_threshold']:.2f}")
        self.sl_obj_th.blockSignals(False)

        self.sl_history.blockSignals(True)
        self.sl_history.setValue(config["history_frames"])
        hf = config["history_frames"]
        self.sl_history_label.setText(str(hf) if hf > 0 else "OFF")
        self.sl_history.blockSignals(False)

    def closeEvent(self, event):
        stop_event.set()
        # キャプチャスレッドの終了を待つ
        if hasattr(self, '_capture_thread') and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=3)
        event.accept()

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

    # MJPEGサーバースレッド（ポート競合時は自動で空きポートへ）
    port = 8080
    for p in [port] + list(range(8081, 8100)):
        try:
            server = HTTPServer(('localhost', p), MJPEGHandler)
            port = p
            break
        except OSError:
            continue
    else:
        print("エラー: 空きポートが見つかりません (8080-8099)")
        sys.exit(1)
    t_server = threading.Thread(target=server.serve_forever, daemon=True)
    t_server.start()
    print(f"ストリーム配信中: http://localhost:{port}/")

    # WebSocket制御サーバー
    ws_port = None
    if HAS_WEBSOCKETS:
        ws_port = start_ws_server(port + 1)
    else:
        print("警告: websocketsがインストールされていないため、WebSocket制御は無効です")

    # PyQt6 GUI（メインスレッド）
    app = QApplication(sys.argv)
    window = MainWindow(cameras)
    window.set_port(port, ws_port)
    window._capture_thread = t_capture

    # WebSocket→GUI同期シグナルを接続
    ws_config_update = window.config_changed_externally

    window.show()
    ret = app.exec()

    # GUI終了後のクリーンアップ
    stop_event.set()
    t_capture.join(timeout=3)
    server.shutdown()
    sys.exit(ret)
