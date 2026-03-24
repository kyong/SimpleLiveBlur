import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# --- 共有フレームバッファ ---
latest_frame = None
frame_lock = threading.Lock()

def blur_faces(frame, detections, padding=0.25):
    h, w = frame.shape[:2]
    result = frame.copy()
    if not detections:
        return result
    for det in detections:
        bb = det.bounding_box
        px = int(bb.width  * padding)
        py = int(bb.height * padding)
        x1 = max(0, bb.origin_x - px)
        y1 = max(0, bb.origin_y - py)
        x2 = min(w, bb.origin_x + bb.width  + px)
        y2 = min(h, bb.origin_y + bb.height + py)
        roi = result[y1:y2, x1:x2]
        if roi.size > 0:
            result[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (99, 99), 30)
    return result

def capture_loop():
    global latest_frame

    # FaceDetector の初期化
    base_options = mp_python.BaseOptions(
        model_asset_path='blaze_face_short_range.tflite'
    )
    options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5
    )
    detector = mp_vision.FaceDetector.create_from_options(options)

    cap = cv2.VideoCapture(0)
    print(f"カメラ: {int(cap.get(3))}x{int(cap.get(4))} @ {cap.get(5):.0f}fps")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        output = blur_faces(frame, result.detections)
        _, jpeg = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with frame_lock:
            latest_frame = jpeg.tobytes()

    cap.release()

# --- MJPEGサーバー ---
class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
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

if __name__ == '__main__':
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()

    port = 8080
    server = HTTPServer(('localhost', port), MJPEGHandler)
    print(f"ストリーム配信中: http://localhost:{port}/")
    print("停止するには Ctrl+C")
    server.serve_forever()