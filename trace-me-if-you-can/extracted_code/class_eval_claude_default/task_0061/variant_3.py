from enum import Enum
from dataclasses import dataclass
from typing import List

class PlayerState(Enum):
    STOPPED = "stopped"
    PLAYING = "playing"

@dataclass
class PlaylistManager:
    songs: List[str]
    
    def add(self, song: str) -> None:
        self.songs.append(song)
    
    def remove(self, song: str) -> bool:
        if song in self.songs:
            self.songs.remove(song)
            return True
        return False
    
    def find_index(self, song: str) -> int:
        try:
            return self.songs.index(song)
        except ValueError:
            return -1
    
    def get_song_at(self, index: int) -> str:
        return self.songs[index] if 0 <= index < len(self.songs) else None
    
    def shuffle_songs(self) -> bool:
        if self.songs:
            import random
            random.shuffle(self.songs)
            return True
        return False

class MusicPlayer:
    def __init__(self):
        self.playlist_manager = PlaylistManager([])
        self.current_song = None
        self.volume = 50

    def add_song(self, song):
        self.playlist_manager.add(song)

    def remove_song(self, song):
        if self.playlist_manager.remove(song) and self.current_song == song:
            self.stop()

    @property
    def playlist(self):
        return self.playlist_manager.songs

    def play(self):
        if self.playlist and self.current_song:
            return self.playlist[0]
        elif len(self.playlist): 
            return False

    def stop(self):
        if self.current_song:
            self.current_song = None
            return True
        return False

    def switch_song(self):
        return self._navigate_relative(1)

    def previous_song(self):
        return self._navigate_relative(-1)

    def _navigate_relative(self, offset):
        if not self.current_song:
            return False
        
        current_idx = self.playlist_manager.find_index(self.current_song)
        if current_idx == -1:
            return False
        
        new_song = self.playlist_manager.get_song_at(current_idx + offset)
        if new_song:
            self.current_song = new_song
            return True
        return False

    def set_volume(self, volume):
        if 0 <= volume <= 100:
            self.volume = volume
        else:
            return False

    def shuffle(self):
        return self.playlist_manager.shuffle_songs()
