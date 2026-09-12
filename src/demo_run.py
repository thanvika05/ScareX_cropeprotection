import sys
import os
import cv2
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.logger import logger
from src.audio.scare_player import scare_player
from src.hardware.hardware import hardware_controller

def run_visual_demo(image_path, output_path):
    logger.info(f"Loading image for demo: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error("Could not load image.")
        return

    logger.info("Loading pre-trained YOLO11n model to simulate the real detection...")
    model = YOLO("yolo11n.pt")
    
    results = model(frame, verbose=False)
    
    bird_detected = False
    best_conf = 0.0
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]
            
            # COCO class for bird is 'bird'
            if class_name.lower() == 'bird' and conf > 0.5:
                bird_detected = True
                if conf > best_conf:
                    best_conf = conf
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                label = f"Crow (Simulated) {conf*100:.1f}%"
                cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # Save the output frame regardless
    cv2.imwrite(output_path, frame)
    logger.info(f"Saved annotated frame to {output_path}")

    if bird_detected:
        logger.info(f"Bird detected with {best_conf*100:.1f}% confidence!")
        logger.info("Triggering Scare Sequence...")
        
        # Trigger hardware actions
        hardware_controller.rotate_camera()
        hardware_controller.flap_wings()
        
        # Play Sound
        scare_player.play_random_scare()
    else:
        logger.info("No bird detected in the image.")

if __name__ == "__main__":
    input_img = r"C:\Users\Ashwanth\.gemini\antigravity-ide\brain\06e819c5-87d9-458b-a77d-c0edaf7b4e1c\demo_bird_1785865417122.png"
    output_img = r"C:\Users\Ashwanth\.gemini\antigravity-ide\brain\06e819c5-87d9-458b-a77d-c0edaf7b4e1c\demo_output.jpg"
    run_visual_demo(input_img, output_img)
