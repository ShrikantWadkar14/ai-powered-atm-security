# main.py
import cv2, time, threading
from queue import Queue
from detectors import PersonWeaponDetector
from tamper import TamperDetector
from action_detector import ActionDetector
from decision import DecisionEngine
from alerts import AlertManager
from flask import Flask
import pkgutil, sys

# VIDEO_SOURCE = 'CCTV Footage of ATM Robbery.mp4' 
# VIDEO_SOURCE = 'cctv_video.mp4' 
# VIDEO_SOURCE = 'Gun store robber shot on camera.mp4' 
# VIDEO_SOURCE = 'test.mp4' 
VIDEO_SOURCE = 'rtsp://admin:admin@'

FRAME_QUEUE_MAX = 500

def frame_reader(src, q):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("ERROR: cannot open video source", src); return
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        if q.full():
            # drop oldest to keep latest
            try:
                q.get_nowait()
            except:
                pass
        q.put(frame)

def detector_worker(q_in, q_out, detector, tamper, action_detector, decision_engine, alert_manager):
    frame_count = 0
    process_every_n = 2  # process heavy detectors every 2 frames
    while True:
        frame = q_in.get()
        frame_count += 1
        # cheap tamper check every frame
        tamper_res = tamper.check(frame)

        persons = []
        weapons = []
        # run object detector every Nth frame to save CPU
        if frame_count % process_every_n == 0:
            objs = detector.predict(frame, imgsz=320, conf=0.35)
            persons = detector.filter_by_class(objs, class_name='person')
            weapons = detector.filter_by_class(objs, class_name='weapon')  # if your weapon model has 'weapon' class
        # tracking / ids handled inside detector (if SORT used)

        # pose/action analysis on cropped persons
        action_res = action_detector.analyze(frame, persons)

        decision = decision_engine.evaluate(persons, weapons, tamper_res, action_res)
        annotated = detector.annotate_frame(frame, persons, weapons, tamper_res, action_res)
        if decision.raise_alert:
            # Save and send annotated frame with detection boxes
            alert_manager.send(decision, annotated)

        # annotate frame for dashboard
        annotated = detector.annotate_frame(frame, persons, weapons, tamper_res, action_res)
        # send annotated to output queue (for Flask streaming)
        if q_out.full():
            try:
                q_out.get_nowait()
            except: pass
        q_out.put(annotated)
        q_in.task_done()

def start_pipeline():
    frame_q = Queue(maxsize=FRAME_QUEUE_MAX)
    out_q = Queue(maxsize=FRAME_QUEUE_MAX)

    
    twilio_cfg = {
        'account_sid': '',
        'auth_token': '',
        'from': '',  # Twilio phone number
        'to': ''     # Destination phone number
    }

    smtp_cfg = {
        'server': 'smtp.gmail.com',
        'port': 465,
        'username': '',
        'password': '',
        'from': '',
        'to': ''
    }

    # instantiate modules
    detector = PersonWeaponDetector(person_model_path="yolov8n.pt", weapon_model_path="best.pt")
    tamper = TamperDetector()
    action_detector = ActionDetector()
    decision_engine = DecisionEngine()
    alert_manager = AlertManager(twilio_cfg=twilio_cfg, smtp_cfg=smtp_cfg)

    # threads
    threading.Thread(target=frame_reader, args=(VIDEO_SOURCE, frame_q), daemon=True).start()
    threading.Thread(target=detector_worker, args=(frame_q, out_q, detector, tamper, action_detector, decision_engine, alert_manager), daemon=True).start()

    return out_q

if __name__ == "__main__":
    print("Starting pipeline...")
    out_q = start_pipeline()

    # minimal CLI viewer if user wants to see frames locally
    while True:
        if not out_q.empty():
            frame = out_q.get()
            cv2.imshow("ATM-POC", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            time.sleep(0.01)
