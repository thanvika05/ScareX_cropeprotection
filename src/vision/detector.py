import os
import cv2
import threading
from ultralytics import YOLO
from src.core.logger import logger
from src.core.config_manager import ConfigManager

class BirdDetector:
    def __init__(self):
        self.config = ConfigManager()
        
        # Try INT8 TFLite first, then float32 TFLite, then PyTorch fallback
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        tflite_int8 = os.path.join(base_dir, 'models', 'yolo11n_int8.tflite')
        tflite_float = os.path.join(base_dir, 'models', 'yolo11n_float32.tflite')
        pt_model = os.path.join(base_dir, 'yolo11n.pt')
        
        if os.path.exists(tflite_int8):
            self.model_path = tflite_int8
        elif os.path.exists(tflite_float):
            self.model_path = tflite_float
        else:
            self.model_path = pt_model
        
        self.model = None
        self.cap = None
        self.is_running = False
        self.thread = None
        
        self.current_frame = None
        self.last_detected_bird = "No Bird Detected"
        self.last_confidence = 0.0

        # Target birds specified in prompt
        self.target_birds = [
            "crow", "common myna", "rose ringed parakeet", "peacock", "pigeon",
            # Add capitalized versions just in case class names differ
            "Crow", "Common Myna", "Rose Ringed Parakeet", "Peacock", "Pigeon"
        ]

        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                logger.info("YOLOv11 model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
        else:
            logger.warning("YOLO model (best.pt) not found. Please train the vision model first.")

    def _vision_loop(self):
        resolution = self.config.get('camera_resolution', [640, 480])
        # Use DirectShow backend on Windows to prevent hanging
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        if not self.cap.isOpened():
            logger.error("Could not open webcam.")
            self.is_running = False
            return

        logger.info("Vision detection started.")
        conf_threshold = self.config.get('camera_confidence', 0.6)

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to grab frame.")
                break

            highest_conf = 0.0
            best_bird = "No Bird Detected"

            if self.model is not None:
                results = self.model(frame, verbose=False)
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = self.model.names[cls]

                        # Check if it's one of our target birds
                        # (If the model is trained ONLY on these birds, we might not need this filter,
                        # but we include it to ignore 'every other object' as requested)
                        if (class_name.lower() in [t.lower() for t in self.target_birds] or class_name.lower() == 'bird') and conf >= conf_threshold:
                            # Draw Bounding Box
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            
                            display_name = class_name if class_name.lower() != 'bird' else 'Bird'
                            label = f"{display_name} {conf*100:.1f}%"
                            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
                            if conf > highest_conf:
                                highest_conf = conf
                                best_bird = display_name
            
            self.last_detected_bird = best_bird
            self.last_confidence = highest_conf
            
            # Convert frame for GUI display (BGR to RGB)
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.cap:
            self.cap.release()

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._vision_loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        self.current_frame = None
        logger.info("Vision detection stopped.")

    def get_latest_detection(self):
        return self.last_detected_bird, self.last_confidence

    def get_current_frame(self):
        return self.current_frame
