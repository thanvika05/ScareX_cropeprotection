import os
import numpy as np
import librosa
import sounddevice as sd
import threading
import time
from src.core.logger import logger
from src.core.config_manager import ConfigManager

try:
    # Try lightweight tflite_runtime first for Raspberry Pi
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # Fallback to full tensorflow
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    except ImportError:
        Interpreter = None
        logger.error("Neither tflite_runtime nor tensorflow found. Audio recognition will fail.")

class AudioRecognizer:
    def __init__(self):
        self.config = ConfigManager()
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'audio_model_int8.tflite')
        
        # Based on user requirements
        self.labels = [
            "Crow", "Common Myna", "Parakeet", "Sparrow", "Dove", "Hen", 
            "Peacock", "Duck", "Goose", "Turkey", "Koel", "Pigeon",
            "background" # For wind, rain, tractor, etc.
        ]
        
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        self.is_running = False
        self.thread = None
        
        self.last_detected_bird = "No Bird Detected"
        self.last_confidence = 0.0

        self.load_model()

    def load_model(self):
        if Interpreter is None:
            return
            
        if os.path.exists(self.model_path):
            try:
                self.interpreter = Interpreter(model_path=self.model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                logger.info("Audio TFLite model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load TFLite audio model: {e}")
        else:
            logger.warning("Audio TFLite model not found. Please run the audio trainer to generate it.")

    def extract_mel_spectrogram(self, audio_data, sample_rate):
        try:
            # Generate Mel Spectrogram
            mel_spect = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128, fmax=8000)
            mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
            
            # Resize to match MobileNetV2 expected input (e.g., 224x224 or 128x128)
            # We'll assume the model expects 128x128x3
            # We pad/truncate along the time axis to get 128
            target_width = 128
            if mel_spect_db.shape[1] < target_width:
                pad_width = target_width - mel_spect_db.shape[1]
                mel_spect_db = np.pad(mel_spect_db, pad_width=((0,0), (0, pad_width)), mode='constant')
            else:
                mel_spect_db = mel_spect_db[:, :target_width]
                
            # Expand to 3 channels by repeating
            mel_spect_rgb = np.stack((mel_spect_db,)*3, axis=-1)
            
            # Normalize to 0-1
            mel_spect_rgb = (mel_spect_rgb - np.min(mel_spect_rgb)) / (np.max(mel_spect_rgb) - np.min(mel_spect_rgb) + 1e-6)
            
            # Ensure FLOAT32 type for TFLite (unless INT8 quantized input is strictly required)
            # If the model is fully INT8 quantized (including input), we'd need to map 0-1 to -128-127
            # For simplicity, assuming the model accepts float32 inputs or we handle it based on input_details
            
            input_type = self.input_details[0]['dtype']
            if input_type == np.int8:
                scale, zero_point = self.input_details[0]['quantization']
                mel_spect_rgb = np.round(mel_spect_rgb / scale + zero_point).astype(np.int8)
            else:
                mel_spect_rgb = mel_spect_rgb.astype(np.float32)
                
            return np.expand_dims(mel_spect_rgb, axis=0) # Add batch dimension
            
        except Exception as e:
            logger.error(f"Mel Spectrogram extraction error: {e}")
            return None

    def _listen_loop(self):
        sample_rate = 22050
        duration = 3  # seconds
        logger.info("Audio recognition (TFLite) started.")
        
        while self.is_running:
            if self.interpreter is None:
                # MOCK MODE: Since no model is available, listen to the mic and trigger on loud sounds
                try:
                    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
                    sd.wait()
                    audio_data = recording.flatten()
                    
                    rms = np.mean(np.abs(audio_data))
                    if rms > 0.05:  # Arbitrary threshold for "loud sound"
                        import random
                        self.last_detected_bird = random.choice(["Crow", "Rose-ringed Parakeet", "Peacock"])
                        self.last_confidence = 0.95
                        logger.info(f"[MOCK AUDIO] Heard a loud sound (RMS: {rms:.3f})! Simulating a {self.last_detected_bird}")
                    else:
                        self.last_detected_bird = "No Bird Detected"
                        self.last_confidence = 0.0
                except Exception as e:
                    logger.error(f"Mock Audio error: {e}")
                    self.last_detected_bird = "Model Missing"
                    time.sleep(2)
                continue
                
            try:
                # Record audio
                recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
                sd.wait()
                
                audio_data = recording.flatten()
                
                features = self.extract_mel_spectrogram(audio_data, sample_rate)
                if features is not None:
                    # Set input tensor
                    self.interpreter.set_tensor(self.input_details[0]['index'], features)
                    # Run inference
                    self.interpreter.invoke()
                    # Get output tensor
                    output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                    
                    # If output is int8, dequantize
                    if self.output_details[0]['dtype'] == np.int8:
                        scale, zero_point = self.output_details[0]['quantization']
                        output_data = (output_data.astype(np.float32) - zero_point) * scale
                        
                    # Model already applies Softmax as the last layer
                    probs = output_data
                    
                    max_prob_index = np.argmax(probs)
                    confidence = probs[max_prob_index]
                    
                    if confidence >= self.config.get('audio_confidence', 0.7):
                        if max_prob_index < len(self.labels):
                            predicted_label = self.labels[max_prob_index]
                            self.last_detected_bird = predicted_label
                        else:
                            self.last_detected_bird = "Unknown"
                        self.last_confidence = confidence
                    else:
                        self.last_detected_bird = "No Bird Detected"
                        self.last_confidence = 0.0
                        
            except Exception as e:
                logger.error(f"Audio recognition error: {e}")
                self.last_detected_bird = "Error"
                time.sleep(1)
                
    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        logger.info("Audio recognition stopped.")

    def get_latest_detection(self):
        return self.last_detected_bird, self.last_confidence
