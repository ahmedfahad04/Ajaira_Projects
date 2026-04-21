class PlaylistManager:
    def __init__(self):
        self.songs = []
        self.currentSong = None
        self.volume = 50

    def add_song(self, song):
        self.songs.append(song)

    def remove_song(self, song):
        if song in self.songs:
            self.songs.remove(song)
            if self.currentSong == song:
                self.stop()

    def play(self):
        if self.songs and self.currentSong:
            return self.songs[0]
        elif len(self.songs): 
            return False

    def stop(self):
        if self.currentSong:
            self.currentSong = None
            return True
        else:
            return False

    def switch_to_next_song(self):
        if self.currentSong:
            current_index = self.songs.index(self.currentSong)
            if current_index < len(self.songs) - 1:
                self.currentSong = self.songs[current_index + 1]
                return True
            else:
                return False
        else:
            return False

    def switch_to_previous_song(self):
        if self.currentSong:
            current_index = self.songs.index(self.currentSong)
            if current_index > 0:
                self.currentSong = self.songs[current_index - 1]
                return True
            else:
                return False
        else:
            return False

    def set_volume(self, volume):
        if 0 <= volume <= 100:
            self.volume = volume
        else:
            return False

    def shuffle(self):
        if self.songs:
            import random
            random.shuffle(self.songs)
            return True
        else:
            return False
