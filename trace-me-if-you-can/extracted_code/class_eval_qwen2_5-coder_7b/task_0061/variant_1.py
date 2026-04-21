class AudioPlayer:
    def __init__(self):
        self.queue = []
        self.currentTrack = None
        self.ampLevel = 50

    def enqueue(self, track):
        self.queue.append(track)

    def dequeue(self, track):
        if track in self.queue:
            self.queue.remove(track)
            if self.currentTrack == track:
                self.halt()

    def start(self):
        if self.queue and self.currentTrack:
            return self.queue[0]
        elif len(self.queue): 
            return False

    def halt(self):
        if self.currentTrack:
            self.currentTrack = None
            return True
        else:
            return False

    def skip(self):
        if self.currentTrack:
            current_index = self.queue.index(self.currentTrack)
            if current_index < len(self.queue) - 1:
                self.currentTrack = self.queue[current_index + 1]
                return True
            else:
                return False
        else:
            return False

    def rewind(self):
        if self.currentTrack:
            current_index = self.queue.index(self.currentTrack)
            if current_index > 0:
                self.currentTrack = self.queue[current_index - 1]
                return True
            else:
                return False
        else:
            return False

    def adjustVolume(self, volume):
        if 0 <= volume <= 100:
            self.ampLevel = volume
        else:
            return False

    def randomize(self):
        if self.queue:
            import random
            random.shuffle(self.queue)
            return True
        else:
            return False
