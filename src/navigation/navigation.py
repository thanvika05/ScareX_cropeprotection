import threading
import time
from src.core.logger import logger
from src.hardware.hardware import hardware_controller

class NavigationEngine:
    def __init__(self):
        self.is_running = False
        self.thread = None
        # Safe distance in meters (e.g., 0.5 meters = 50cm)
        self.safe_distance = 0.5 

    def _navigate_loop(self):
        logger.info("Navigation engine started. Robot is roaming.")
        while self.is_running:
            distance = hardware_controller.get_distance()
            
            if distance < self.safe_distance:
                logger.info(f"Obstacle detected at {distance:.2f}m. Evading...")
                hardware_controller.stop()
                time.sleep(0.5)
                hardware_controller.move_backward()
                time.sleep(1.0)
                hardware_controller.turn_right()
                time.sleep(1.0)
                hardware_controller.stop()
            else:
                # Slowly move forward or roam
                hardware_controller.move_forward()
            
            time.sleep(0.2)
            
    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._navigate_loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        hardware_controller.stop()
        logger.info("Navigation engine stopped.")

navigation_engine = NavigationEngine()
