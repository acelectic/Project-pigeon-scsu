import RPi.GPIO as GPIO
import time


import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args.accumulate(args.integers))

# Pin Definitons:
led_pin = 12  # BOARD pin 12


def main():
    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
    GPIO.setup(led_pin, GPIO.OUT)  # LED pin set as output

    # Initial state for LEDs:
    GPIO.output(led_pin, GPIO.LOW)
    print("Starting demo now! Press CTRL+C to exit")
    try:
        state = True
        while True:
            if state:
                GPIO.output(led_pin, GPIO.HIGH)
                state = False
            else:
                GPIO.output(led_pin, GPIO.LOW)
                state = True
            time.sleep(5)
    finally:
        GPIO.cleanup()  # cleanup all GPIO


if __name__ == '__main__':
    main()
