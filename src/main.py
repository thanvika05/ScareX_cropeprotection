import time
import sys
from src.core.logger import logger
from src.vision.detector import BirdDetector
from src.audio.recognizer import AudioRecognizer
from src.fusion.fusion_engine import FusionEngine
from src.navigation.navigation import navigation_engine
from src.dashboard.app import start_flask_app

def main():
    logger.info("Initializing ScareX v2.0...")
    
    try:
        # Initialize Core Modules
        vision_module = BirdDetector()
        audio_module = AudioRecognizer()
        
        # Initialize Fusion Engine (combines Vision + Audio, controls Alarms)
        fusion_engine = FusionEngine(vision_module, audio_module)
        
        # Start Dashboard in a background thread
        start_flask_app(fusion_engine, vision_module, audio_module)
        
        # Start AI modules
        vision_module.start()
        audio_module.start()
        
        # Start Engines
        fusion_engine.start()
        navigation_engine.start()
        
        logger.info("All systems started. Press Ctrl+C to stop.")
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down ScareX...")
    except Exception as e:
        logger.error(f"Critical error: {e}")
    finally:
        try:
            navigation_engine.stop()
            fusion_engine.stop()
            vision_module.stop()
            audio_module.stop()
        except:
            pass
        logger.info("Shutdown complete.")
        sys.exit(0)

if __name__ == "__main__":
    main()
