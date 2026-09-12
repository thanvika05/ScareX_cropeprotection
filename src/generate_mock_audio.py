import numpy as np
from scipy.io.wavfile import write
import os

def create_synthetic_wav(filename, frequency=440, duration=2.0, is_noise=False):
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    if is_noise:
        # Generate random white noise for background
        wave = np.random.normal(0, 0.1, len(t))
    else:
        # Generate a sine wave for mock birds
        wave = 0.5 * np.sin(2 * np.pi * frequency * t)
        
    # Convert to 16-bit PCM
    wave_16bit = np.int16(wave * 32767)
    write(filename, sample_rate, wave_16bit)
    print(f"Created {filename}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    scare_dir = os.path.join(base_dir, 'dataset', 'audio', 'ScareCalls')
    bird_dir = os.path.join(base_dir, 'dataset', 'audio', 'birds', 'crow')
    noise_dir = os.path.join(base_dir, 'dataset', 'audio', 'birds', 'background')
    
    os.makedirs(scare_dir, exist_ok=True)
    os.makedirs(bird_dir, exist_ok=True)
    os.makedirs(noise_dir, exist_ok=True)
    
    create_synthetic_wav(os.path.join(scare_dir, 'mock_scare1.wav'), frequency=300, duration=1.0)
    create_synthetic_wav(os.path.join(scare_dir, 'mock_scare2.wav'), frequency=600, duration=1.0)
    create_synthetic_wav(os.path.join(bird_dir, 'mock_crow1.wav'), frequency=800, duration=1.0)
    create_synthetic_wav(os.path.join(bird_dir, 'mock_crow2.wav'), frequency=800, duration=1.0)
    
    # Generate background noise so the model doesn't predict "crow" for silence
    for i in range(5):
        create_synthetic_wav(os.path.join(noise_dir, f'noise_{i}.wav'), duration=1.0, is_noise=True)
