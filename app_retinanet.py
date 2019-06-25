import os, time, cv2
from flask import Flask, render_template, Response, jsonify, request, make_response, redirect, url_for
from multiprocessing import Process, Value, Pool
from importlib import import_module
import requests


import time, glob
# from imutils.video import VideoStream


# from until.yolo_model import yolo_model
# yolo = yolo_model()

if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from until.camera_opencv import Camera

# load yolo model
from until.detect import yolo_detect

app = Flask(__name__)
status = False
confidence = 0.5
detect_every_frame = 2

# class model_config:
#     def __init__(self, confidence, detect_every_frame):
#         self.__confidence = confidence
#         self.__detect_every_frame = detect_every_frame
#         print('ss')
#
#     def setConfidence(self, confident):
#         self.__confidence = float(confident)
#
#     def getConfidence(self):
#         return self.__confidence
#
#     def setDetect_every_frame(self, detect_every_frame):
#         self.__detect_every_frame = int(detect_every_frame)
#
#     def getDetect_every_frame(self):
#         return self.__detect_every_frame
#
# modelcf = model_config(confidence=0.7, detect_every_frame=5)

# vdo4test = glob.glob('video/*.mp4')
# def get_my_ip():
#     return jsonify({'ip': request.remote_addr}), 200

@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/', methods=['POST', 'GET'])
def index():
    """Video streaming home page."""
    ip = request.remote_addr
    if request.method == 'POST':
        data = request.form
        if data['snap'] == "go2snap":
            return redirect(url_for('snap', ip=ip))


    print(request.remote_addr)
    return render_template('index.html', status = 'on' if status else 'off', _pass='pigeon' , confidence= confidence ,detect_every_frame=detect_every_frame, ip=ip, fullip = "http://{}:5000/".format(ip))


def gen(camera = None):
    while True:
        # # cap =
        # # time.sleep(1)
        # # _, img = cv2.VideoCapture(0).read()
        # img = VideoStream(0).read()
        # # cap.release()
        # frame = cv2.imencode('.jpg', img)[1].tobytes()
        frame = camera.get_frame()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()), mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/mode/<path:subpath>', methods=["POST"])
def mode(subpath):
    import time
    global p, status
    # print(request.subject)

    if subpath == 'off':
        if status:
            p.terminate()
            print('Detect ON')
            status = False
            # return "Detect OFF"
        # else:
            # return 'Detect is already OFF'


    elif subpath == 'on':
        if status:
            # return 'Detect is already ON'
            pass
        else:
            status = True
            p = Process(target=run, args=())
            p.start()
            # return 'Detect ON'


    return redirect(url_for('index'))

@app.route('/set/<path:subpath>', methods=["POST"])
def set(subpath):
    global detect_every_frame, confidence
    # print(request.form['frame'])
    try :
        data = request.get_json(force=True)
    except:
        data = request.form
    if subpath == 'frame':
        detect_every_frame = data['frame']
        # print(modelcf.getConfidence(), modelcf.getDetect_every_frame())
    elif subpath == 'confidence':
        confidence = data['confidence']
        # print(modelcf.getConfidence(), modelcf.getDetect_every_frame())

    return redirect(url_for('index'))

@app.route('/snap', methods=["GET"])
def snap(ip = '127.0.0.1'):
    print('from',ip)
    return render_template('live.html', ip=ip)

def run():
    from retinanet import retinanet_model
    retinanet = retinanet_model.Model(confidence=confidence)
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('retinanet/video/YouTube5.mp4')

    frameindex = 1
    while True:

        _, frame = cap.read()

        # if _:
        #     out = retinanet.detect(frame)
        #     cv2.imshow('ss', out)
        #     if cv2.waitKey(25) & 0xFF == ord('q'):
        #         break
        #     print("loop running")

        # if _  and (frameindex // detect_every_frame == 0 or detect_every_frame == 1):
        #     out = retinanet.detect(frame)
        #     cv2.imshow('ss', out)
        #     if cv2.waitKey(25) & 0xFF == ord('q'):
        #         break
        #     print("loop running")

        if _  and frameindex % int(detect_every_frame) == 0:
            out = retinanet.detect(frame)
            cv2.imshow('ss', out)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            print("loop running")

        frameindex += 1
    cap.release()

    global status
    status = False
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    # test()
    # fulltest()
