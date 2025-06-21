"""
Sound Notification System
Plays audio notifications when development sessions are completed.
"""

import os
import subprocess
import tempfile
import math
import wave
import struct
from typing import Optional


class SoundNotifier:
    def __init__(self):
        self.audio_available = self._check_audio_availability()
    
    def _check_audio_availability(self) -> bool:
        """Check if audio playback is available on the system"""
        try:
            # Check for common Linux audio players
            for cmd in ['paplay', 'aplay', 'play']:
                result = subprocess.run(['which', cmd], capture_output=True)
                if result.returncode == 0:
                    return True
            return False
        except Exception:
            return False
    
    def _generate_completion_tone(self, filename: str, duration: float = 1.0):
        """Generate a pleasant completion tone WAV file"""
        sample_rate = 44100
        frames = int(duration * sample_rate)
        
        # Create a pleasant two-tone chime
        tone1_freq = 800  # Higher tone
        tone2_freq = 600  # Lower tone
        
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            for i in range(frames):
                # Create envelope to avoid clicks
                envelope = 1.0
                if i < sample_rate * 0.05:  # Fade in
                    envelope = i / (sample_rate * 0.05)
                elif i > frames - sample_rate * 0.05:  # Fade out
                    envelope = (frames - i) / (sample_rate * 0.05)
                
                # First half: higher tone, second half: lower tone
                freq = tone1_freq if i < frames // 2 else tone2_freq
                
                # Generate sine wave
                t = i / sample_rate
                amplitude = 0.3 * envelope  # 30% volume with envelope
                sample = amplitude * math.sin(2 * math.pi * freq * t)
                
                # Convert to 16-bit integer
                sample_int = int(sample * 32767)
                wav_file.writeframes(struct.pack('<h', sample_int))
    
    def _play_system_sound(self, sound_file: str) -> bool:
        """Play sound using system audio commands"""
        try:
            # Try different audio players
            players = [
                ['paplay', sound_file],
                ['aplay', sound_file],
                ['play', sound_file]
            ]
            
            for player_cmd in players:
                try:
                    result = subprocess.run(player_cmd, capture_output=True, timeout=5)
                    if result.returncode == 0:
                        return True
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
            
            return False
        except Exception as e:
            print(f"Error playing sound: {e}")
            return False
    
    def play_completion_sound(self) -> bool:
        """Play a completion notification sound"""
        if not self.audio_available:
            print("🔊 Audio not available - task completed!")
            return False
        
        try:
            # Create temporary sound file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_filename = temp_file.name
            
            # Generate and play completion tone
            self._generate_completion_tone(temp_filename)
            success = self._play_system_sound(temp_filename)
            
            # Clean up
            try:
                os.unlink(temp_filename)
            except Exception:
                pass
            
            if success:
                print("🔊 Task completed!")
            else:
                print("🔊 Audio playback failed - task completed!")
            
            return success
            
        except Exception as e:
            print(f"🔊 Sound notification error: {e} - task completed!")
            return False
    
    def play_custom_sound(self, sound_file: str) -> bool:
        """Play a custom sound file"""
        if not self.audio_available:
            print("🔊 Audio not available")
            return False
        
        if not os.path.exists(sound_file):
            print(f"🔊 Sound file not found: {sound_file}")
            return False
        
        return self._play_system_sound(sound_file)


# Global instance
notifier = SoundNotifier()


def play_completion_notification() -> bool:
    """Convenience function to play completion sound"""
    return notifier.play_completion_sound()


def play_custom_notification(sound_file: str) -> bool:
    """Convenience function to play custom sound"""
    return notifier.play_custom_sound(sound_file)


if __name__ == "__main__":
    # Test the notification system
    print("Testing completion sound...")
    play_completion_notification()