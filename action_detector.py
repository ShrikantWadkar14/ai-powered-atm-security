import mediapipe as mp
import cv2
import time
from collections import defaultdict, deque

mp_pose = mp.solutions.pose

class ActionDetector:
    def __init__(self, loiter_seconds=60):
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        self.track_times = defaultdict(lambda: time.time())
        self.loiter_seconds = loiter_seconds
        # naive store of previous centers to get motion magnitude (idless)
        self.prev_centers = {}
        self.motion_deques = defaultdict(lambda: deque(maxlen=5))

    def analyze(self, frame, persons):
        """
        persons: list of {'box':(x1,y1,x2,y2), ...}
        returns: dict with keys 'actions' (list) and 'loitering' True/False
        """
        actions = []
        now = time.time()
        loitering_flag = False
        for i, p in enumerate(persons):
            x1,y1,x2,y2 = p['box']
            h,w = frame.shape[:2]
            # clamp cropping
            x1c, y1c = max(0,x1), max(0,y1)
            x2c, y2c = min(w-1,x2), min(h-1,y2)
            if x2c-x1c < 20 or y2c-y1c < 20:
                continue
            crop = frame[y1c:y2c, x1c:x2c]
            # pose on crop (fast)
            res = self.pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            fell = False
            if res.pose_landmarks:
                # simple fall detection: large vertical movement of nose or torso over short time
                # (placeholder heuristics; tune after tests)
                lm = res.pose_landmarks.landmark
                # use nose y normalized
                nose_y = lm[mp_pose.PoseLandmark.NOSE].y
                # push into motion history (naive)
                center = ((x1+x2)//2, (y1+y2)//2)
                prev = self.prev_centers.get(i)
                if prev:
                    motion = ((center[0]-prev[0])**2 + (center[1]-prev[1])**2)**0.5
                else:
                    motion = 0.0
                self.prev_centers[i] = center
                self.motion_deques[i].append(motion)
                # if motion high and bounding box center low in frame -> possible fall/violent
                if sum(self.motion_deques[i]) / len(self.motion_deques[i]) > 40.0:
                    actions.append({'id': i, 'type': 'violent_motion'})
                # loose faint detection: nose very low in crop (body collapsed)
                if nose_y > 0.8:
                    actions.append({'id': i, 'type': 'possible_faint'})
            # loiter: update timestamp when detected
            # naive loiter: if appears continuously > threshold
            if i not in self.track_times:
                self.track_times[i] = now
            duration = now - self.track_times[i]
            if duration > self.loiter_seconds:
                loitering_flag = True
        return {'actions': actions, 'loitering': loitering_flag}
