import os
import random
from src.core.logger import logger
from src.core.config_manager import ConfigManager
import pygame

class AlertManager:
    def __init__(self):
        self.config = ConfigManager()
        self.audio_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets', 'audio')
        
        # 5 Required Deterrent Sounds
        self.deterrent_sounds = {
            "Changeable Hawk-Eagle": "hawk_eagle.mp3",
            "Long-eared Owl": "owl.mp3",
            "European Pied Flycatcher": "flycatcher.mp3",
            "Stresemann's Bushcrow": "bushcrow.mp3",
            "Red-vented Bulbul": "bulbul.mp3"
        }
        
        self.unplayed = []
        self.last_played = None
        self._fill_bag()
        
        pygame.mixer.init()
        self.set_volume(self.config.get('speaker_volume', 1.0))

    def _fill_bag(self):
        new_bag = list(self.deterrent_sounds.keys())
        random.shuffle(new_bag)
        
        # Ensure the first sound doesn't match the last played sound from previous cycle
        if self.last_played and new_bag[0] == self.last_played:
            if len(new_bag) > 1:
                new_bag[0], new_bag[1] = new_bag[1], new_bag[0]
                
        self.unplayed = new_bag

    def set_volume(self, volume):
        pygame.mixer.music.set_volume(max(0.0, min(1.0, volume)))

    def play_alert(self, bird_name, retry_count=0):
        # Prevent infinite recursion if all files are missing/corrupted
        if retry_count >= len(self.deterrent_sounds):
            logger.error("[ALERT] All sounds failed or missing. Cannot play deterrent.")
            return "Error"
            
        if not self.unplayed:
            self._fill_bag()
            
        action = self.unplayed.pop(0)
        filename = self.deterrent_sounds[action]
        file_path = os.path.join(self.audio_dir, filename)
        
        if os.path.exists(file_path):
            try:
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                logger.info(f"[ALERT] Playing '{action}' for {bird_name}")
                self.last_played = action
                return action
            except Exception as e:
                logger.error(f"[ALERT] Error playing {filename}: {e}")
                return self.play_alert(bird_name, retry_count + 1)
        else:
            logger.warning(f"[ALERT] Sound file not found: {file_path}. Skipping.")
            return self.play_alert(bird_name, retry_count + 1)

    def stop(self):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            logger.info("[ALERT] Stopped playing.")

alert_manager = AlertManager()
