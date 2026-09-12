import json
import os
import logging

class ConfigManager:
    _instance = None
    _config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'config.json')

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance.config = cls._instance.load_config()
        return cls._instance

    def load_config(self):
        try:
            if os.path.exists(self._config_path):
                with open(self._config_path, 'r') as f:
                    return json.load(f)
            else:
                logging.warning(f"Config file not found at {self._config_path}, using defaults.")
                return self.get_default_config()
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
            return self.get_default_config()

    def save_config(self):
        try:
            os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
            with open(self._config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logging.info("Configuration saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save config: {e}")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self.save_config()

    def get_default_config(self):
        return {
            "camera_confidence": 0.6,
            "audio_confidence": 0.7,
            "detection_delay_sec": 5.0,
            "scare_duration_sec": 10.0,
            "speaker_volume": 1.0,
            "camera_resolution": [640, 480]
        }
