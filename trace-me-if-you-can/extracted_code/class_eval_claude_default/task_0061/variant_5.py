class MusicPlayer:
    def __init__(self):
        self.playlist = []
        self.current_song = None
        self.volume = 50

    def add_song(self, song):
        self.playlist.append(song)

    def remove_song(self, song):
        song_exists = song in self.playlist
        if song_exists:
            self.playlist.remove(song)
            self._handle_current_song_removal(song)

    def _handle_current_song_removal(self, removed_song):
        if self.current_song == removed_song:
            self.stop()

    def play(self):
        return {
            (True, True): self.playlist[0],
            (True, False): False,
            (False, True): None,
            (False, False): None
        }.get((bool(self.playlist), bool(self.current_song)))

    def stop(self):
        was_playing = self.current_song is not None
        self.current_song = None
        return was_playing

    def switch_song(self):
        return self._change_song(lambda idx: idx + 1, lambda idx, length: idx < length - 1)

    def previous_song(self):
        return self._change_song(lambda idx: idx - 1, lambda idx, length: idx > 0)

    def _change_song(self, index_transform, boundary_check):
        if not self.current_song:
            return False
        
        try:
            current_index = self.playlist.index(self.current_song)
            if boundary_check(current_index, len(self.playlist)):
                new_index = index_transform(current_index)
                self.current_song = self.playlist[new_index]
                return True
        except (ValueError, IndexError):
            pass
        
        return False

    def set_volume(self, volume):
        is_valid_volume = 0 <= volume <= 100
        if is_valid_volume:
            self.volume = volume
        return False if not is_valid_volume else None

    def shuffle(self):
        has_songs = len(self.playlist) > 0
        if has_songs:
            import random
            random.shuffle(self.playlist)
        return has_songs
