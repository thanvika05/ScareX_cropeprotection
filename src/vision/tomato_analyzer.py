import cv2
import time
from ultralytics import YOLO

class TomatoAnalyzer:
    def __init__(self, weights_path="yolov8x/weights/best.pt"):
        self.weights_path = weights_path
        self.model = YOLO(self.weights_path)
        self.cap = None
        self.running = False
        self.session_stats = {'fully_ripened': 0, 'green': 0, 'half_ripened': 0}
        self.frame_count = 0
        self.start_time = 0
        self.current_frame = None
        self.class_names = {0: 'fully_ripened', 1: 'green', 2: 'half_ripened'}
        
    def start(self, source):
        self.stop()
        self.session_stats = {'fully_ripened': 0, 'green': 0, 'half_ripened': 0}
        self.frame_count = 0
        self.start_time = time.time()
        
        # Determine source
        if str(source).isdigit():
            source_val = int(source)
        else:
            source_val = source
            
        self.cap = cv2.VideoCapture(source_val)
        if not self.cap.isOpened():
            print(f"Error: Could not open source {source_val}.")
            return False
            
        self.running = True
        return True
        
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
            
    def get_frame(self):
        if not self.running or not self.cap:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return None
            
        self.frame_count += 1
        
        # Run inference
        results = self.model.predict(frame, conf=0.60, iou=0.45, verbose=False)
        result = results[0]
        
        # Temporary frame detections to avoid double counting easily
        frame_detections = set()
        
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = self.class_names.get(cls_id, f"Class {cls_id}")
            
            # Very basic approach: just sum frame detections to get total observations (as per prompt)
            self.session_stats[class_name] += 1
            
            # Draw
            color = (0, 255, 0) if cls_id == 1 else (0, 0, 255) if cls_id == 0 else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
        self.current_frame = frame
        return frame
