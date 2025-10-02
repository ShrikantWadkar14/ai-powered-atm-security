from flask import Flask, render_template, Response, request, jsonify
import cv2
import threading
from queue import Queue
import os
from werkzeug.utils import secure_filename
from detectors import PersonWeaponDetector
from tamper import TamperDetector
from action_detector import ActionDetector
from decision import DecisionEngine
from alerts import AlertManager

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

frame_queue = None
pipeline_thread = None
stop_flag = False
recent_alerts = []

class AlertCollector:
    def __init__(self):
        self.alerts = []
        self.last_alert_times = {}  # {(level, tuple(reasons)): timestamp}

    def add_alert(self, level, reasons, status="Sent", snapshot_path=None):
        import time
        key = (level, tuple(reasons))
        now = time.time()
        last_time = self.last_alert_times.get(key, 0)
        if now - last_time < 30:
            # Don't add duplicate alert within 30 seconds
            return
        self.last_alert_times[key] = now
        self.alerts.append({
            "level": level,
            "reasons": reasons,
            "status": status,
            "timestamp": now,
            "snapshot_path": snapshot_path
        })
alert_collector = AlertCollector()

twilio_cfg = {
    'account_sid': '',
    'auth_token': '',
    'from': '',
    'to': ''
}
smtp_cfg = {
    'server': 'smtp.gmail.com',
    'port': 465,
    'username': '',
    'password': '',
    'from': '',
    'to': ''
}
alert_manager = AlertManager(twilio_cfg=twilio_cfg, smtp_cfg=smtp_cfg)

def generate_frames():
    global frame_queue, stop_flag
    frame_count = 0
    import time
    last_frame_time = time.time()
    while not stop_flag:
        if frame_queue and not frame_queue.empty():
            frame = frame_queue.get()
            frame_count += 1
            # Stream only every 2nd frame for speed
            if frame_count % 2 != 0:
                continue
            frame = cv2.resize(frame, (640, 360))
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ret:
                frame_bytes = buffer.tobytes()
                print(f"Streaming frame at {time.strftime('%H:%M:%S')}")
                last_frame_time = time.time()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # If no frame available, send a blank image every 2 seconds
            if time.time() - last_frame_time > 2:
                import numpy as np
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                ret, buffer = cv2.imencode('.jpg', blank, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                last_frame_time = time.time()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if not pipeline_thread or not pipeline_thread.is_alive():
        # Detection not running, return empty response
        return Response(b'', mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global frame_queue, pipeline_thread, stop_flag
    if pipeline_thread and pipeline_thread.is_alive():
        return jsonify({"error": "Detection already running"}), 400
    source_type = request.form.get('source_type')
    if source_type == 'file':
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"error": "No video file selected"}), 400
        filename = secure_filename(video_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(filepath)
        video_source = filepath
    else:  # RTSP
        rtsp_url = request.form.get('rtsp_url')
        if not rtsp_url:
            return jsonify({"error": "No RTSP URL provided"}), 400
        video_source = rtsp_url
    stop_flag = False
    frame_queue = Queue(maxsize=500)
    pipeline_thread = threading.Thread(target=start_detection_pipeline, 
                                    args=(video_source, frame_queue, alert_collector))
    pipeline_thread.daemon = True
    pipeline_thread.start()
    return jsonify({"message": "Detection started"})

def start_detection_pipeline(video_source, frame_queue, alert_collector):
    print(f"[INFO] Starting detection pipeline with source: {video_source}")
    detector = PersonWeaponDetector(person_model_path="yolov8n.pt", weapon_model_path="best.pt")
    tamper = TamperDetector()
    action_detector = ActionDetector()
    decision_engine = DecisionEngine()
    alert_manager = AlertManager(twilio_cfg=twilio_cfg, smtp_cfg=smtp_cfg)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {video_source}")
        return
    frame_idx = 0
    while not stop_flag:
        ret, frame = cap.read()
        frame_idx += 1
        if not ret:
            print(f"[WARN] Frame not read at index {frame_idx}. End of stream or error.")
            break
        print(f"[INFO] Processing frame {frame_idx}")
        tamper_res = tamper.check(frame)
        objs = detector.predict(frame, imgsz=320, conf=0.35)
        persons = detector.filter_by_class(objs, class_name='person')
        weapons = detector.filter_by_class(objs, class_name='weapon')
        action_res = action_detector.analyze(frame, persons)
        decision = decision_engine.evaluate(persons, weapons, tamper_res, action_res)
        annotated = detector.annotate_frame(frame, persons, weapons, tamper_res, action_res)
        if decision.raise_alert:
            snap_path = alert_manager.save_snapshot(annotated, decision)
            alert_collector.add_alert(decision.level, decision.reasons, snapshot_path=snap_path)
            alert_manager.send(decision, annotated)
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except Exception as e:
                print(f"[WARN] Could not remove frame from full queue: {e}")
        frame_queue.put(annotated)
        print(f"[INFO] Frame {frame_idx} put into queue.")
    cap.release()
    print(f"[INFO] Detection pipeline stopped. Released video source.")

@app.route('/latest_alert_snapshot')
def latest_alert_snapshot():
    # Return the latest alert snapshot image
    if not alert_collector.alerts:
        return '', 404
    snap_path = alert_collector.alerts[-1].get('snapshot_path')
    if not snap_path or not os.path.exists(snap_path):
        return '', 404
    return Response(open(snap_path, 'rb').read(), mimetype='image/jpeg')

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global stop_flag
    stop_flag = True
    return jsonify({"message": "Detection stopped"})

@app.route('/get_alerts')
def get_alerts():
    return jsonify(alert_collector.alerts)

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)