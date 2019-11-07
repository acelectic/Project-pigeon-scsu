from playsound import playsound

class Silen_control:
    def __init__(self):
        self.silen_file = 'fast.wav'
    def run_alert(self):
        print('alert')
        playsound(self.silen_file)