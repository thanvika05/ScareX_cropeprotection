import logging
import os
from datetime import datetime

def setup_logger():
    # Define log file path
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Use current date for log file
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f'scarex_{date_str}.log')

    # Create logger
    logger = logging.getLogger("ScareX")
    logger.setLevel(logging.DEBUG)

    # Prevent adding duplicate handlers if setup is called multiple times
    if not logger.handlers:
        # Create file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter and add it to the handlers
        # Formatting exactly as requested: Date Time Event...
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

# Create a global logger instance
logger = setup_logger()

def log_event(event_type, details):
    """
    Utility function for structured logging of events like detections or scares.
    Example: log_event("DETECTION", "Bird: Crow, Confidence: 85%")
    """
    logger.info(f"[{event_type}] {details}")
