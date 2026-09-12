import os
import shutil
from ultralytics import YOLO
from src.core.logger import logger

def train_vision_model():
    """
    Train YOLOv11 on the birds dataset and save the best model.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_yaml_path = os.path.join(base_dir, 'dataset', 'images', 'BIRDS.v1i.yolov11', 'data.yaml')
    models_dir = os.path.join(base_dir, 'models')

    if not os.path.exists(data_yaml_path):
        logger.error(f"data.yaml not found at {data_yaml_path}")
        return False

    logger.info("Starting YOLOv11 training...")
    try:
        # Load a pre-trained YOLOv11 nano model
        # Assuming yolo11n.pt is available or will be downloaded by ultralytics
        model = YOLO('yolo11n.pt') 
        
        # Train the model
        results = model.train(
            data=data_yaml_path,
            epochs=1,
            imgsz=640,
            project=os.path.join(base_dir, 'logs', 'yolo_training'),
            name='bird_detection'
        )
        
        # Find the best.pt from the training run
        # ultralytics saves it in project/name/weights/best.pt
        best_pt_src = os.path.join(base_dir, 'logs', 'yolo_training', 'bird_detection', 'weights', 'best.pt')
        best_pt_dst = os.path.join(models_dir, 'best.pt')
        
        if os.path.exists(best_pt_src):
            os.makedirs(models_dir, exist_ok=True)
            shutil.copy2(best_pt_src, best_pt_dst)
            logger.info(f"Model successfully trained and saved to {best_pt_dst}")
            return True
        else:
            logger.error(f"best.pt not found at expected location: {best_pt_src}")
            return False

    except Exception as e:
        logger.error(f"Error during YOLO training: {e}")
        return False

if __name__ == "__main__":
    train_vision_model()
