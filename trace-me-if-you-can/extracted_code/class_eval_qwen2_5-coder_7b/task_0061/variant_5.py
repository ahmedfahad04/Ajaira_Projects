class MusicControl:
    def __init__(self):
        self.trackList = []
        self.currentTrack = None
        self.volume = 50

    def addTrack(self, track):
        self.trackList.append(track)

    def removeTrack(self, track):
        if track in self.trackList:
            self.trackList.remove(track)
            if self.currentTrack == track:
                self.stop()

    def startPlayback(self):
        if self.trackList and self.currentTrack:
            return self.trackList[0]
        elif len(self.trackList): 
            return False

    def stopPlayback(self):
        if self.currentTrack:
            self.currentTrack = None
            return True
        else:
            return False

    def skipToNextTrack(self):
        if self.currentTrack:
            current_index = self.trackList.index(self.currentTrack)
            if current_index < len(self.trackList) - 1:
                self.currentTrack = self.trackList[current_index + 1]
                return True
            else:
                return False
        else:
            return False

    def skipToPreviousTrack(self):
        if self.currentTrack:
            current_index = self.trackList.index(self.currentTrack)
            if current_index > 0:
                self.currentTrack = self.trackList[current_index - 1]
                return True
            else:
                return False
        else:
            return False

    def adjustVolume(self, volume):
        if 0 <= volume <= 100:
            self.volume = volume
        else:
            return False

    def randomizeTracks(self):
        if self.trackList:
            import random
            random.shuffle(self.trackList)
            return True
        else:
            return False
