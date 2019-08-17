# SDA = pin.SDA_1
# SCL = pin.SCL_1
# SDA_1 = pin.SDA
# SCL_1 = pin.SCL

from adafruit_servokit import ServoKit
import board
import busio
import time

# On the Jetson Nano
# Bus 0 (pins 28,27) is board SCL_1, SDA_1 in the jetson board definition file
# Bus 1 (pins 5, 3) is board SCL, SDA in the jetson definition file
# Default is to Bus 1; We are using Bus 0, so we need to construct the busio first ...


# kit[0] is the bottom servo
# kit[1] is the top servo



class camera_control():

    def __init__(self):
        self.default_angle = 90

        self.vertical_angle = self.default_angle
        self.horizontal_angle = self.default_angle
        self.step_angle = 5

        self.max_angle = 170
        self.min_angle = 10

        print("Initializing Servos")
        self.i2c_bus0 = (busio.I2C(board.SCL_1, board.SDA_1))
        print("Initializing ServoKit")
        self.kit = ServoKit(channels=16, i2c=self.i2c_bus0)

        self._setDefault()

    def __moveVertical(self, angle):
        self.kit.servo[0].angle = angle
        self.vertical_angle = angle
        time.sleep(0.5)

    def __moveHorizontal(self, angle):
        self.kit.servo[1].angle = angle
        self.horizontal_angle = angle
        time.sleep(0.5)

    def _setDefault(self):
        self.__moveHorizontal(angle=self.default_angle)
        time.sleep(1)
        self.__moveVertical(angle=self.default_angle)

    def rotateToDefault(self):
        self.__moveHorizontal(angle=self.default_angle)
        time.sleep(1)
        self.__moveVertical(angle=self.default_angle)
        time.sleep(1)

    def rotateLeft(self):
        next_angle = self.vertical_angle + self.step_angle
        if  next_angle > self.min_angle:
            self.__moveVertical(next_angle)


    def rotateRight(self):
        next_angle = self.vertical_angle - self.step_angle
        if  next_angle < self.max_angle:
            self.__moveVertical(next_angle)

    def rotateDown(self):
        next_angle = self.horizontal_angle + self.step_angle
        if  next_angle > self.min_angle:
            self.__moveHorizontal(next_angle)


    def rotateUp(self):
        next_angle = self.horizontal_angle - self.step_angle
        if  next_angle < self.max_angle:
            self.__moveHorizontal(next_angle)

servo = camera_control()

servo.rotateToDefault()

for i in range(20):
    servo.rotateRight()
    time.sleep(1)

for i in range(20):
    servo.rotateLeft()
    time.sleep(1)

for i in range(20):
    servo.rotateUp()
    time.sleep(1)

for i in range(20):
    servo.rotateDown()
    time.sleep(1)

servo.rotateToDefault()