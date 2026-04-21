class MusicPlayer:
    VOLUME_MIN, VOLUME_MAX = 0, 100
    DEFAULT_VOLUME = 50

    def __init__(self):
        self.playlist = []
        self.current_song = None
        self.volume = self.DEFAULT_VOLUME

    def add_song(self, song):
        self.playlist.append(song)

    def remove_song(self, song):
        if song in self.playlist:
            self.playlist.remove(song)
            if self.current_song == song:
                self.stop()

    def play(self):
        has_playlist = bool(self.playlist)
        has_current = bool(self.current_song)
        
        if has_playlist and has_current:
            return self.playlist[0]
        elif has_playlist:
            return False

    def stop(self):
        currently_playing = self.current_song is not None
        self.current_song = None
        return currently_playing

    def switch_song(self):
        next_index = self._calculate_next_index(1)
        return self._set_song_by_index(next_index)

    def previous_song(self):
        prev_index = self._calculate_next_index(-1)
        return self._set_song_by_index(prev_index)

    def _calculate_next_index(self, step):
        if not self.current_song or self.current_song not in self.playlist:
            return None
        
        current_index = self.playlist.index(self.current_song)
        candidate_index = current_index + step
        
        return candidate_index if self._is_valid_index(candidate_index) else None

    def _is_valid_index(self, index):
        return 0 <= index < len(self.playlist)

    def _set_song_by_index(self, index):
        if index is not None:
            self.current_song = self.playlist[index]
            return True
        return False

    def set_volume(self, volume):
        volume_valid = self.VOLUME_MIN <= volume <= self.VOLUME_MAX
        if volume_valid:
            self.volume = volume
        else:
            return False

    def shuffle(self):
        playlist_exists = bool(self.playlist)
        if playlist_exists:
            import random
            random.shuffle(self.playlist)
        return playlist_exists
