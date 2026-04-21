import random
from typing import List, Optional

class MusicPlayer:
    def __init__(self):
        self._playlist: List[str] = []
        self._current_song: Optional[str] = None
        self._volume: int = 50

    def add_song(self, song: str) -> None:
        self._playlist.append(song)

    def remove_song(self, song: str) -> None:
        try:
            self._playlist.remove(song)
            if self._current_song == song:
                self.stop()
        except ValueError:
            pass

    def play(self):
        if not self._playlist:
            return None
        return self._playlist[0] if self._current_song else False

    def stop(self) -> bool:
        was_playing = self._current_song is not None
        self._current_song = None
        return was_playing

    def switch_song(self) -> bool:
        return self._navigate_playlist(1)

    def previous_song(self) -> bool:
        return self._navigate_playlist(-1)

    def _navigate_playlist(self, direction: int) -> bool:
        if not self._current_song or not self._playlist:
            return False
        
        try:
            current_index = self._playlist.index(self._current_song)
            new_index = current_index + direction
            
            if 0 <= new_index < len(self._playlist):
                self._current_song = self._playlist[new_index]
                return True
        except ValueError:
            pass
        
        return False

    def set_volume(self, volume: int) -> Optional[bool]:
        if 0 <= volume <= 100:
            self._volume = volume
            return None
        return False

    def shuffle(self) -> bool:
        if self._playlist:
            random.shuffle(self._playlist)
            return True
        return False
