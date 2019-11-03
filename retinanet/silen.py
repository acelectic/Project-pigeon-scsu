from playsound import playsound

class Silen_control:
    def __init__(self):
        self.silen_file = 'fast.wav'
    def alert(self):
        playsound(self.silen_file)