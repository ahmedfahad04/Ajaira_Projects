class MusicPlayer:
    def __init__(self):
        self.playlist = []
        self.current_song = None
        self.volume = 50
        self._current_index = -1

    def add_song(self, song):
        self.playlist.append(song)

    def remove_song(self, song):
        if song not in self.playlist:
            return
        
        song_index = self.playlist.index(song)
        self.playlist.remove(song)
        
        if self.current_song == song:
            self.stop()
        elif self._current_index >= song_index:
            self._current_index -= 1

    def play(self):
        if self.playlist and self.current_song:
            return self.playlist[0]
        elif len(self.playlist): 
            return False

    def stop(self):
        result = bool(self.current_song)
        self.current_song = None
        self._current_index = -1
        return result

    def switch_song(self):
        return self._move_to_song(self._get_current_index() + 1)

    def previous_song(self):
        return self._move_to_song(self._get_current_index() - 1)

    def _get_current_index(self):
        if self.current_song and self.current_song in self.playlist:
            return self.playlist.index(self.current_song)
        return -1

    def _move_to_song(self, target_index):
        if not self.current_song or target_index < 0 or target_index >= len(self.playlist):
            return False
        
        self.current_song = self.playlist[target_index]
        self._current_index = target_index
        return True

    def set_volume(self, volume):
        if 0 <= volume <= 100:
            self.volume = volume
        else:
            return False

    def shuffle(self):
        if not self.playlist:
            return False
        
        import random
        random.shuffle(self.playlist)
        return True
