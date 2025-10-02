import cv2

class TamperDetector:
    def __init__(self, mean_black_thresh=10, mean_white_thresh=245, std_blur_thresh=8,
                 freeze_diff_thresh=2, freeze_duration=3, fps=25):
        self.prev_gray = None
        self.freeze_counter = 0
        self.mean_black_thresh = mean_black_thresh
        self.mean_white_thresh = mean_white_thresh
        self.std_blur_thresh = std_blur_thresh
        self.freeze_diff_thresh = freeze_diff_thresh
        self.freeze_limit = int(fps * freeze_duration)   # convert seconds frames

    def check(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        std = float(gray.std())

        if mean < self.mean_black_thresh:
            return {'covered': True, 'reason': 'black_frame'}
        if mean > self.mean_white_thresh:
            return {'covered': True, 'reason': 'white_frame'}
        if std < self.std_blur_thresh:
            return {'covered': True, 'reason': 'blurred'}

        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            avg_diff = float(diff.mean())
            if avg_diff < self.freeze_diff_thresh:
                self.freeze_counter += 1
            else:
                self.freeze_counter = 0
            if self.freeze_counter >= self.freeze_limit:
                return {'covered': True, 'reason': 'frozen_frame'}

        self.prev_gray = gray
        return {'covered': False}
