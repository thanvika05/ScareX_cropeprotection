import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.logger import logger
from src.core.decision_engine import DecisionEngine
from src.vision.detector import BirdDetector
from src.audio.recognizer import AudioRecognizer

def test_system():
    logger.info("--- STARTING AUTOMATED HEADLESS TEST ---")
    
    # Initialize components
    vision_module = BirdDetector()
    audio_module = AudioRecognizer()
    engine = DecisionEngine(vision_module, audio_module)
    
    # Start the engine
    engine.start()
    
    logger.info("Simulating an Audio detection (Confidence 0.95)")
    engine.activate_scare_mode("Parrot", 0.95)
    
    # Let it run for a few seconds to let hardware actions and sounds play
    time.sleep(5)
    
    logger.info("Simulating a Vision detection (Confidence 0.88)")
    engine.activate_scare_mode("common_myna", 0.88)
    
    time.sleep(5)
    
    # Cleanup
    engine.stop()
    logger.info("--- AUTOMATED TEST COMPLETE ---")

if __name__ == "__main__":
    test_system()
