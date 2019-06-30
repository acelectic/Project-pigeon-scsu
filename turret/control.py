import Jetson.GPIO as GPIO


class Turret():

    def __init__(self):
        self.cameraMovestep = 5;
        self.areaConfirm = .05, 20  #.05 percent of width and height, min size = 20px

    def moveCamera(self):
        pass

    def calDistant2target(self, centroid, frame_shape):
        return