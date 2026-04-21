class SoundPlayer:
    def __init__(self):
        self.library = []
        self.currentSong = None
        self.soundLevel = 50

    def addSong(self, song):
        self.library.append(song)

    def deleteSong(self, song):
        if song in self.library:
            self.library.remove(song)
            if self.currentSong == song:
                self.stop()

    def playSong(self):
        if self.library and self.currentSong:
            return self.library[0]
        elif len(self.library): 
            return False

    def stop(self):
        if self.currentSong:
            self.currentSong = None
            return True
        else:
            return False

    def changeSong(self):
        if self.currentSong:
            current_index = self.library.index(self.currentSong)
            if current_index < len(self.library) - 1:
                self.currentSong = self.library[current_index + 1]
                return True
            else:
                return False
        else:
            return False

    def goBack(self):
        if self.currentSong:
            current_index = self.library.index(self.currentSong)
            if current_index > 0:
                self.currentSong = self.library[current_index - 1]
                return True
            else:
                return False
        else:
            return False

    def adjustSound(self, volume):
        if 0 <= volume <= 100:
            self.soundLevel = volume
        else:
            return False

    def randomizeSongs(self):
        if self.library:
            import random
            random.shuffle(self.library)
            return True
        else:
            return False
