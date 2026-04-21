class TunePlayer:
    def __init__(self):
        self.songs = []
        self.currentTune = None
        self.volumeLevel = 50

    def addTune(self, tune):
        self.songs.append(tune)

    def removeTune(self, tune):
        if tune in self.songs:
            self.songs.remove(tune)
            if self.currentTune == tune:
                self.stopPlaying()

    def playTune(self):
        if self.songs and self.currentTune:
            return self.songs[0]
        elif len(self.songs): 
            return False

    def stopPlaying(self):
        if self.currentTune:
            self.currentTune = None
            return True
        else:
            return False

    def nextTune(self):
        if self.currentTune:
            current_index = self.songs.index(self.currentTune)
            if current_index < len(self.songs) - 1:
                self.currentTune = self.songs[current_index + 1]
                return True
            else:
                return False
        else:
            return False

    def previousTune(self):
        if self.currentTune:
            current_index = self.songs.index(self.currentTune)
            if current_index > 0:
                self.currentTune = self.songs[current_index - 1]
                return True
            else:
                return False
        else:
            return False

    def adjustVolume(self, volume):
        if 0 <= volume <= 100:
            self.volumeLevel = volume
        else:
            return False

    def shuffleTunes(self):
        if self.songs:
            import random
            random.shuffle(self.songs)
            return True
        else:
            return False
