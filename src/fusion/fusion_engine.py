import threading
import time
from src.core.logger import logger
from src.logging.db_logger import db_logger
from src.core.config_manager import ConfigManager
from src.hardware.hardware import hardware_controller
from src.alerts.alert_manager import alert_manager

class FusionEngine:
    def __init__(self, vision_module, audio_module):
        self.vision = vision_module
        self.audio = audio_module
        self.config = ConfigManager()
        
        self.is_running = False
        self.thread = None
        self.last_scare_time = 0
        self.scare_active = False
        self.consecutive_detections = 0

    def _engine_loop(self):
        logger.info("Fusion engine started.")
        while self.is_running:
            v_bird, v_conf = self.vision.get_latest_detection()
            a_bird, a_conf = self.audio.get_latest_detection()

            current_time = time.time()
            deterrent_cooldown = self.config.get('deterrent_cooldown', 10.0)
            required_consecutive = self.config.get('required_consecutive_detections', 3)
            
            # Check thresholds
            v_valid = (v_bird != "No Bird Detected" and v_conf >= self.config.get('camera_confidence', 0.6))
            a_valid = (a_bird != "No Bird Detected" and a_bird not in ["Error", "background", "Model Missing"] and a_conf >= self.config.get('audio_confidence', 0.7))

            if not self.scare_active and (current_time - self.last_scare_time) > deterrent_cooldown:
                if v_valid or a_valid:
                    self.consecutive_detections += 1
                else:
                    self.consecutive_detections = 0

                if self.consecutive_detections >= required_consecutive:
                    if v_valid and a_valid:
                        # Both detected: High confidence! Use vision for exact species if different, or fuse
                        final_bird = v_bird if v_conf >= a_conf else a_bird
                        final_conf = max(v_conf, a_conf) + 0.05 # slight boost for fusion
                        logger.info(f"Fusion Match! Both vision and audio detected a bird: {final_bird}")
                        self.trigger_scare(final_bird, final_conf, "FUSION")
                        self.consecutive_detections = 0
                    
                    elif v_valid and not a_valid:
                        # Vision only
                        self.trigger_scare(v_bird, v_conf, "CAMERA")
                        self.consecutive_detections = 0
                        
                    elif a_valid and not v_valid:
                        # Audio only - Rotate camera to investigate
                        logger.info(f"Audio detected {a_bird} ({a_conf:.2f}), but no vision confirmation. Rotating camera...")
                        hardware_controller.rotate_camera()
                        time.sleep(1.0) # wait for camera to settle
                        
                        # Check vision again
                        v_bird2, v_conf2 = self.vision.get_latest_detection()
                        if v_bird2 != "No Bird Detected" and v_conf2 >= self.config.get('camera_confidence', 0.5):
                            # Confirmed visually
                            logger.info("Visual confirmation successful after rotation!")
                            self.trigger_scare(v_bird2, max(v_conf2, a_conf), "FUSION")
                        else:
                            # Only audio, still trigger
                            logger.info("No visual confirmation, triggering based on audio anyway.")
                            self.trigger_scare(a_bird, a_conf, "MICROPHONE")
                        self.consecutive_detections = 0
            else:
                self.consecutive_detections = 0

            time.sleep(0.5)

    def trigger_scare(self, bird_name, confidence, source="UNKNOWN"):
        self.scare_active = True
        self.last_scare_time = time.time()
        scare_duration = self.config.get('scare_duration_sec', 10.0)
        
        # Play alarm
        alarm_played = alert_manager.play_alert(bird_name)
        
        # Hardware action (e.g. flap wings)
        threading.Thread(target=hardware_controller.execute_scare_sequence).start()
        
        # Log to DB
        db_logger.log_event("SCARE_ACTIVATED", species=bird_name, confidence=confidence, alarm_played=alarm_played, system_status="SCARE", detection_source=source)

        # Schedule deactivate
        threading.Timer(scare_duration, self.deactivate_scare).start()

    def deactivate_scare(self):
        alert_manager.stop()
        hardware_controller.stop()
        self.scare_active = False
        logger.info("Scare mode deactivated. System ready.")

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._engine_loop, daemon=True)
            self.thread.start()
            db_logger.log_event("SYSTEM_START", system_status="RUNNING")

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.deactivate_scare()
        logger.info("Fusion engine stopped.")
        db_logger.log_event("SYSTEM_STOP", system_status="STOPPED")
