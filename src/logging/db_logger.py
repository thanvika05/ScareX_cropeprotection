import sqlite3
import os
import threading
from datetime import datetime
from src.core.logger import logger
from src.core.config_manager import ConfigManager

class DBLogger:
    def __init__(self):
        self.config = ConfigManager()
        self.db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), self.config.get('db_path', 'scarex.db'))
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.lock = threading.Lock()
        self.init_db()

    def init_db(self):
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        event_type TEXT NOT NULL,
                        species TEXT,
                        confidence REAL,
                        alarm_played TEXT,
                        system_status TEXT
                    )
                ''')
                # Check if detection_source column exists, add it if not
                cursor.execute("PRAGMA table_info(events)")
                columns = [info[1] for info in cursor.fetchall()]
                if 'detection_source' not in columns:
                    cursor.execute('ALTER TABLE events ADD COLUMN detection_source TEXT')
                
                # Tomato monitoring columns
                if 'duration' not in columns:
                    cursor.execute('ALTER TABLE events ADD COLUMN duration REAL')
                if 'green_count' not in columns:
                    cursor.execute('ALTER TABLE events ADD COLUMN green_count INTEGER')
                if 'half_ripened_count' not in columns:
                    cursor.execute('ALTER TABLE events ADD COLUMN half_ripened_count INTEGER')
                if 'fully_ripened_count' not in columns:
                    cursor.execute('ALTER TABLE events ADD COLUMN fully_ripened_count INTEGER')
                if 'harvest_priority' not in columns:
                    cursor.execute('ALTER TABLE events ADD COLUMN harvest_priority TEXT')
                
                conn.commit()
                conn.close()
            logger.info("SQLite Database initialized with tomato monitoring columns.")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite Database: {e}")

    def log_event(self, event_type, species=None, confidence=None, alarm_played=None, system_status=None, detection_source="UNKNOWN", duration=None, green_count=None, half_ripened_count=None, fully_ripened_count=None, harvest_priority=None):
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO events (event_type, species, confidence, alarm_played, system_status, detection_source, duration, green_count, half_ripened_count, fully_ripened_count, harvest_priority)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (event_type, species, confidence, alarm_played, system_status, detection_source, duration, green_count, half_ripened_count, fully_ripened_count, harvest_priority))
                conn.commit()
                conn.close()
            # Also log to file/console
            details = f"Species: {species}, Status: {system_status}, Source: {detection_source}"
            logger.info(f"[{event_type}] {details}")
        except Exception as e:
            logger.error(f"Failed to log event to DB: {e}")

    def get_recent_events(self, limit=50):
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM events ORDER BY timestamp DESC LIMIT ?', (limit,))
                rows = cursor.fetchall()
                conn.close()
                return rows
        except Exception as e:
            logger.error(f"Failed to retrieve events from DB: {e}")
            return []

db_logger = DBLogger()
