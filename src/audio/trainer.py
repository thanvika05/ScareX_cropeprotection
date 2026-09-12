import os
import imageio_ffmpeg
# Make FFmpeg available to librosa for decoding mp3s
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

import glob
import numpy as np
import librosa
from src.core.logger import logger
import tensorflow as tf
from tensorflow.keras import layers, models, applications

class AudioTrainer:
    def __init__(self):
        self.dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'dataset', 'audio')
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.labels = [
            "Crow", "Common Myna", "Parakeet", "Sparrow", "Dove", "Hen", 
            "Peacock", "Duck", "Goose", "Turkey", "Koel", "Pigeon",
            "background"
        ]

    def _extract_mel_features(self, file_path, sample_rate=22050, target_width=128):
        try:
            audio, _ = librosa.load(file_path, sr=sample_rate, duration=3.0)
            mel_spect = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, fmax=8000)
            mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
            
            if mel_spect_db.shape[1] < target_width:
                pad_width = target_width - mel_spect_db.shape[1]
                mel_spect_db = np.pad(mel_spect_db, pad_width=((0,0), (0, pad_width)), mode='constant')
            else:
                mel_spect_db = mel_spect_db[:, :target_width]
                
            mel_spect_rgb = np.stack((mel_spect_db,)*3, axis=-1)
            # Normalize
            mel_spect_rgb = (mel_spect_rgb - np.min(mel_spect_rgb)) / (np.max(mel_spect_rgb) - np.min(mel_spect_rgb) + 1e-6)
            return mel_spect_rgb
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None

    def _load_data(self):
        X, y = [], []
        logger.info(f"Loading dataset from {self.dataset_dir}")
        for label_idx, label in enumerate(self.labels):
            folder = os.path.join(self.dataset_dir, label)
            if not os.path.exists(folder):
                continue
                
            files = glob.glob(os.path.join(folder, '*.wav'))
            files.extend(glob.glob(os.path.join(folder, '*.mp3')))
            for file in files:
                features = self._extract_mel_features(file)
                if features is not None:
                    X.append(features)
                    y.append(label_idx)
                    
        return np.array(X), np.array(y)

    def train_and_export(self):
        X, y = self._load_data()
        
        # Shuffle data to prevent validation split issues
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        if len(X) == 0:
            logger.error("No training data found! Please add .wav files to dataset/audio/<label>/")
            return
            
        logger.info(f"Loaded {len(X)} samples. Building MobileNetV2 model...")
        
        # Build model using pre-trained weights for transfer learning
        base_model = applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.labels), activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        logger.info("Training model...")
        # Train (in a real scenario, split into train/val)
        model.fit(X, y, epochs=25, batch_size=8, validation_split=0.2)
        
        # Export to TFLite with INT8 quantization
        logger.info("Exporting to TFLite INT8 format...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Generator for representative dataset to enable INT8 quantization
        def representative_data_gen():
            for i in range(min(100, len(X))):
                yield [np.expand_dims(X[i].astype(np.float32), axis=0)]
                
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_quant_model = converter.convert()
        
        out_path = os.path.join(self.models_dir, 'audio_model_int8.tflite')
        with open(out_path, 'wb') as f:
            f.write(tflite_quant_model)
            
        logger.info(f"Model saved to {out_path}")

if __name__ == "__main__":
    trainer = AudioTrainer()
    trainer.train_and_export()
