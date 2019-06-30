import Jetson.GPIO as GPIO
import optparse
import time

parser = optparse.OptionParser()

parser.add_option('-n', '--num',
    action="store", dest="num",
    help="query string", default="spam")

options, args = parser.parse_args()

print ('Query string:', options.num, args, options)

GPIO.setmode(GPIO.BOARD)

mode = GPIO.getmode()
print(mode)

GPIO.setup(options.num, GPIO.OUT, initial=GPIO.LOW)

for i in range(10):

    if i/2 == 0:
        GPIO.output(options.num, GPIO.HIGH)
    else:
        GPIO.output(options.num, GPIO.LOW)
    time.sleep(1)

GPIO.cleanup()
# class Turret():
#
#     def __init__(self):
#         self.cameraMovestep = 5;
#         self.areaConfirm = .05, 20  #.05 percent of width and height, min size = 20px
#
#     def moveCamera(self):
#         pass
#
#     def calDistant2target(self, centroid, frame_shape):
#         return