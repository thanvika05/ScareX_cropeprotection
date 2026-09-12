import os
import cv2
import numpy as np

def create_mock_yolo_dataset():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(base_dir, 'dataset', 'images', 'BIRDS.v1i.yolov11')
    
    # Create directories for images and labels
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(dataset_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, split, 'labels'), exist_ok=True)
    
    # Create a couple of mock 640x640 images and label files
    for i in range(2):
        img_path = os.path.join(dataset_dir, 'train', 'images', f'mock_bird_{i}.jpg')
        label_path = os.path.join(dataset_dir, 'train', 'labels', f'mock_bird_{i}.txt')
        
        # Generate random noise image
        img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        # Draw a fake bird (a red rectangle)
        cv2.rectangle(img, (200, 200), (400, 400), (0, 0, 255), -1)
        cv2.imwrite(img_path, img)
        
        # Write YOLO label (class_id x_center y_center width height)
        # Assuming Crow is class 0, center is 0.5, 0.5, size is ~0.3, 0.3
        with open(label_path, 'w') as f:
            f.write("0 0.468 0.468 0.312 0.312\n")

    # Copy one to valid for validation
    cv2.imwrite(os.path.join(dataset_dir, 'valid', 'images', 'mock_bird_val.jpg'), img)
    with open(os.path.join(dataset_dir, 'valid', 'labels', 'mock_bird_val.txt'), 'w') as f:
        f.write("0 0.468 0.468 0.312 0.312\n")

    # Create data.yaml
    yaml_content = f"""
train: {os.path.join(dataset_dir, 'train', 'images')}
val: {os.path.join(dataset_dir, 'valid', 'images')}
test: {os.path.join(dataset_dir, 'test', 'images')}

nc: 5
names: ['Crow', 'Common Myna', 'Rose Ringed Parakeet', 'Peacock', 'Pigeon']
"""
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    print("Mock YOLO dataset created at:", dataset_dir)

if __name__ == "__main__":
    create_mock_yolo_dataset()
