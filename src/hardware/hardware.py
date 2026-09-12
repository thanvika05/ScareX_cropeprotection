import time
from src.core.logger import logger
from src.core.config_manager import ConfigManager

try:
    from gpiozero import Motor, DistanceSensor, Servo
    GPIO_AVAILABLE = True
except ImportError:
    logger.warning("gpiozero not available. Using Mock Hardware.")
    GPIO_AVAILABLE = False

class MockMotor:
    def forward(self): logger.info("[MOCK] Motor moving forward")
    def backward(self): logger.info("[MOCK] Motor moving backward")
    def stop(self): logger.info("[MOCK] Motor stopped")

class MockDistanceSensor:
    @property
    def distance(self): return 1.0 # Returns 1 meter

class MockServo:
    def max(self): logger.info("[MOCK] Servo MAX")
    def min(self): logger.info("[MOCK] Servo MIN")
    def mid(self): logger.info("[MOCK] Servo MID")
    def detach(self): pass

class RobotHardware:
    def __init__(self):
        self.config = ConfigManager()
        pins = self.config.get("gpio_pins", {})
        
        self.is_active = False

        if GPIO_AVAILABLE:
            try:
                # Motor 1 (Left)
                self.motor_left = Motor(forward=pins.get("motor_in1", 17), backward=pins.get("motor_in2", 27))
                # Motor 2 (Right)
                self.motor_right = Motor(forward=pins.get("motor_in3", 22), backward=pins.get("motor_in4", 23))
                # Ultrasonic Sensor
                self.ultrasonic = DistanceSensor(echo=pins.get("ultrasonic_echo", 25), trigger=pins.get("ultrasonic_trigger", 24))
                # Camera Servo
                self.camera_servo = Servo(pins.get("servo_camera", 18))
                logger.info("Hardware initialized via gpiozero.")
            except Exception as e:
                logger.error(f"Error initializing GPIO: {e}. Falling back to Mock.")
                self._init_mock()
        else:
            self._init_mock()

    def _init_mock(self):
        self.motor_left = MockMotor()
        self.motor_right = MockMotor()
        self.ultrasonic = MockDistanceSensor()
        self.camera_servo = MockServo()

    def move_forward(self):
        logger.info("[HARDWARE] Move Forward")
        self.motor_left.forward()
        self.motor_right.forward()

    def move_backward(self):
        logger.info("[HARDWARE] Move Backward")
        self.motor_left.backward()
        self.motor_right.backward()

    def turn_left(self):
        logger.info("[HARDWARE] Turn Left")
        self.motor_left.backward()
        self.motor_right.forward()

    def turn_right(self):
        logger.info("[HARDWARE] Turn Right")
        self.motor_left.forward()
        self.motor_right.backward()

    def stop(self):
        logger.info("[HARDWARE] Stop")
        self.motor_left.stop()
        self.motor_right.stop()

    def rotate_camera(self):
        logger.info("[HARDWARE] Rotating Camera")
        self.camera_servo.max()
        time.sleep(1)
        self.camera_servo.min()
        time.sleep(1)
        self.camera_servo.mid()
        # To avoid jitter
        if hasattr(self.camera_servo, 'detach'):
            self.camera_servo.detach()
            
    def get_distance(self):
        dist = self.ultrasonic.distance
        return dist

    def execute_scare_sequence(self):
        self.is_active = True
        # Move randomly or flash lights (if any), flap wings (simulated by turning left/right)
        self.turn_left()
        time.sleep(0.5)
        self.turn_right()
        time.sleep(0.5)
        self.stop()
        self.is_active = False

hardware_controller = RobotHardware()
