from ultralytics import YOLO
import numpy as np
import cv2

class PersonWeaponDetector:
    def __init__(self, person_model_path="yolov8n.pt", weapon_model_path="best.pt", device='cpu'):
        # If you have separate weapon model, you can load another YOLO model
        self.person_model = YOLO(person_model_path)
        self.weapon_model = None
        if weapon_model_path:
            self.weapon_model = YOLO(weapon_model_path)

    def predict(self, frame, imgsz=320, conf=0.35):
        objs = []
        # Run person model and only keep class 0 (person)
        results_person = self.person_model(frame, imgsz=imgsz, conf=conf)[0]
        for b in results_person.boxes:
            cls = int(b.cls[0])
            if cls == 0:  # Only person class
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf_score = float(b.conf[0])
                objs.append({'box': (x1, y1, x2, y2), 'score': conf_score, 'cls': cls})

        # Run weapon model if available
        if self.weapon_model:
            results_weapon = self.weapon_model(frame, imgsz=imgsz, conf=conf)[0]
            for b in results_weapon.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf_score = float(b.conf[0])
                # Only accept weapon detections above threshold and not overlapping with person boxes
                weapon_conf_thresh = 0.5
                if conf_score < weapon_conf_thresh:
                    continue
                weapon_box = (x1, y1, x2, y2)
                # Check overlap with person boxes
                overlap = False
                for p in objs:
                    if p['cls'] == 0:
                        px1, py1, px2, py2 = p['box']
                        # Calculate intersection over union (IoU)
                        ix1 = max(x1, px1)
                        iy1 = max(y1, py1)
                        ix2 = min(x2, px2)
                        iy2 = min(y2, py2)
                        iw = max(0, ix2 - ix1)
                        ih = max(0, iy2 - iy1)
                        inter = iw * ih
                        area_w = (x2 - x1) * (y2 - y1)
                        area_p = (px2 - px1) * (py2 - py1)
                        union = area_w + area_p - inter
                        iou = inter / union if union > 0 else 0
                        if iou > 0.3:
                            overlap = True
                            break
                if not overlap:
                    objs.append({'box': weapon_box, 'score': conf_score, 'cls': 'weapon'})
        return objs

    def filter_by_class(self, objs, class_name='person'):
        if class_name == 'person':
            return [o for o in objs if o['cls'] == 0]
        if class_name == 'weapon':
            return [o for o in objs if o['cls'] == 'weapon']
        return []

    def annotate_frame(self, frame, persons, weapons, tamper_res, action_res):
        out = frame.copy()
        # persons
        for p in persons:
            x1,y1,x2,y2 = p['box']
            cv2.rectangle(out, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(out, f"Person {p.get('score',0):.2f}", (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
        # weapons
        for w in weapons:
            x1,y1,x2,y2 = w['box']
            cv2.rectangle(out, (x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(out, f"Weapon {w.get('score',0):.2f}", (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        # tamper overlay
        if tamper_res.get('covered'):
            cv2.putText(out, "TAMPER DETECTED: "+tamper_res.get('reason',''), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255),2)
        # actions
        if action_res.get('loitering'):
            cv2.putText(out, "LOITERING", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255),2)
        return out
