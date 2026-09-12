import cv2
import time
import argparse
from ultralytics import YOLO

# Define class names mapping based on the provided labels
CLASS_NAMES = {
    0: 'b_fully_ripened',
    1: 'b_green',
    2: 'b_half_ripened'
}

def main():
    parser = argparse.ArgumentParser(description="Real-time Tomato Maturity Detection")
    parser.add_argument("--conf", type=float, default=0.60, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--source", type=str, default="0", help="Webcam source ID or video file path")
    parser.add_argument("--weights", type=str, default="yolov8x/weights/best.pt", help="Path to model weights")
    args = parser.parse_args()

    import sys
    import os
    # Add current directory to path so src can be imported
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.logging.db_logger import db_logger

    # Parse source correctly
    source_val = args.source
    if source_val.isdigit():
        source_val = int(source_val)

    # Load the YOLOv8 model
    print(f"Loading model from {args.weights}...")
    try:
        model = YOLO(args.weights)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Open video source
    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        print(f"Error: Could not open source {source_val}.")
        return

    print(f"Starting inference on {source_val}... Press 'q' to quit.")

    # For FPS calculation
    prev_time = 0
    
    # Session tracking
    session_detections = {0: 0, 1: 0, 2: 0} # b_fully_ripened, b_green, b_half_ripened
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or failed to capture image. Stopping gracefully...")
            break

        # Run inference
        results = model.predict(frame, conf=args.conf, iou=args.iou, verbose=False)
        
        # Calculate FPS
        new_time = time.time()
        fps = 1 / (new_time - prev_time) if prev_time > 0 else 0
        prev_time = new_time
        
        # Process results and draw bounding boxes
        result = results[0]
        frame_count += 1
        for box in result.boxes:
            # Extract box coordinates, confidence, and class id
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Tally
            if cls_id in session_detections:
                session_detections[cls_id] += 1
                
            # Map class ID to name
            class_name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
            label = f"{class_name} {conf:.2f}"
            
            # Draw bounding box
            # Use different colors for different classes
            color = (0, 255, 0) if cls_id == 1 else (0, 0, 255) if cls_id == 0 else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Display FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow("Tomato Maturity Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Save the monitoring session
    print("Saving monitoring session...")
    summary = f"Processed {frame_count} frames. Detections: {session_detections[0]} fully ripened, {session_detections[1]} green, {session_detections[2]} half ripened."
    total = sum(session_detections.values())
    full_pct = (session_detections[0] / total * 100) if total > 0 else 0
    priority = "LOW"
    if full_pct > 50:
        priority = "HIGH"
    elif full_pct > 20:
        priority = "MEDIUM"

    db_logger.log_event(
        event_type="MONITORING_SESSION_END",
        species="Tomato",
        confidence=1.0,
        alarm_played="None",
        system_status=summary,
        detection_source=str(source_val),
        duration=0,
        green_count=session_detections[1],
        half_ripened_count=session_detections[2],
        fully_ripened_count=session_detections[0],
        harvest_priority=priority
    )
    print("Session saved successfully.")

if __name__ == "__main__":
    main()
