
from until.elas_api import Elas_api
import retinanet.set_model2environ
import os
from importlib import import_module
from multiprocessing import Process

import cv2
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, render_template, Response, request, redirect, url_for, make_response
from tzlocal import get_localzone

if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from until.camera_opencv import Camera


try:

    from until.camera_control import camera_control
    cam_api = camera_control()
except:
    print("can't connect servo")


es_ip = '192.168.1.29'
es_port = 9200
es = Elas_api(ip=es_ip)
es_status = None
es_status = es.checkStatus()[0]

app = Flask(__name__)

status = False
confidence = 0.5
sec_per_frame = 5


@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/', methods=['POST', 'GET'])
def index(ips=None):
    """Video streaming home page."""
    ip = request.remote_addr
    if request.method == 'POST':

        try:
            data = request.form
            print(data)
            if data['live'] == "live_camera":
                return redirect(url_for('live_camera', ip=ip))
        except:
            pass
    ips = request.args.get('ip')
    print(ips)
    return render_template('index.html', status='on' if status else 'off', _pass='pigeon', confidence=confidence,
                           sec_per_frame=sec_per_frame, ip=ip, fullip="http://{}:5000/".format(ips), es_status=es_status,
                           es_ip=es_ip + ':{}'.format(es_port))


def gen(camera=None):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/mode/<path:subpath>', methods=["POST"])
def mode(subpath):
    global p, status
    # print(request.subject)

    if subpath == 'off':
        if status:
            p.terminate()
            status = False
            print('Detect Off')
        # else:
        # return 'Detect is already OFF'

    elif subpath == 'on':
        if status:
            # return 'Detect is already ON'
            pass
        else:
            status = True
            p = Process(target=run, args=())
            # p = Process(target=runtest, args=())

            p.start()
            p.join()
            print('Detect On')

    return make_response('toggle mode to ' + subpath)


@app.route('/detectStatus', methods=["POST"])
def getDetectStatus():
    global status
    print('check status: {}'.format('on' if status else 'off'))
    return make_response('on' if status else 'off')


@app.route('/set/<path:subpath>', methods=["POST"])
def set(subpath):
    global sec_per_frame, confidence
    # print(request.form['frame'])
    try:
        data = request.get_json(force=True)
    except:
        data = request.form

    print(data)
    if subpath == 'frame':
        sec_per_frame = data['frame']
        print('frame', sec_per_frame)

    elif subpath == 'confidence':
        confidence = data['confidence']
        print('confidencw', confidence)

    return redirect(url_for('index'))


@app.route('/live_camera', methods=["GET"])
def live_camera(ip='127.0.0.1'):
    print('from', ip)
    return render_template('live.html', ip=ip)


@app.route('/camera_command', methods=["POST"])
def camera_command():
    try:
        data = request.get_json(force=True)
    except:
        data = request.form

    print(data)
    cmd = data['cmd']
    if cmd == 'Left':
        cam_api.rotateLeft()
        print(cmd)
    elif cmd == 'Right':
        cam_api.rotateRight()
        print(cmd)
    elif cmd == 'Up':
        cam_api.rotateUp()
        print(cmd)
    elif cmd == 'Down':
        cam_api.rotateDown()
        print(cmd)
    elif cmd == 'Default':
        cam_api.rotateToDefault()
        print(cmd)

    headers = {"Content-Type": "application/json"}

    return make_response("rotate camera " + cmd)
    # return redirect(url_for('snap'))


def run(vdo_=0):
    cam_api = None

    def _a(es, cam_api):
        global status_detect, shot_status
        print("{:#^20}{}{:#^20}\nconfidence:{}\nSec per frame{}".format(
            '', 'Detect ON', '', confidence, sec_per_frame))
        from retinanet import retinanet_model
        from datetime import datetime

        retinanet = retinanet_model.Model(
            confidence=confidence, es=es, es_mode=True, cam_api=cam_api, model_is='c_resnet50')

        status_detect = False
        # vdo_ = 'video/YouTube4.mp4'
        # cap = cv2.VideoCapture(vdo_)
        # vdo_ = 'video/video_25620705_061211.mp4'
        cap = cv2.VideoCapture(0)

        def task_deley():
            global status_detect
            status_detect = True
            print('Detect status: {}'.format(status_detect))

        scheduler = BackgroundScheduler(timezone=get_localzone())
        scheduler.add_job(task_deley, 'interval', seconds=sec_per_frame)
        scheduler.start()

        while True:
            _, frame = cap.read()
            if _:
                if status_detect:
                    status_detect = False
                    print("loop running")
                    r = retinanet.detect(frame)
                    
        cap.release()

    _a(es, cam_api)
    global status
    status = False
    # return redirect(url_for('index'))
    return make_response('detect off')


# def runtest(vdo_=0):
#     cam_api = None

#     def _a(es, cam_api):
#         global status_detect, shot_status
#         print("{:#^20}{}{:#^20}\nconfidence:{}\nSec per frame{}".format(
#             '', 'Detect ON', '', confidence, sec_per_frame))
#         from retinanet import retinanet_model
#         from datetime import datetime

#         retinanet = retinanet_model.Model(
#             confidence=confidence, es=es, es_mode=True, cam_api=cam_api, model_is='c_resnet50')

#         status_detect = False
#         shot_status = False
#         # vdo_ = 'video/YouTube4.mp4'
#         # cap = cv2.VideoCapture(vdo_)
#         vdo_ = 'video/video_25620705_061211.mp4'
#         cap = cv2.VideoCapture(vdo_)

#         def task_deley():
#             global status_detect
#             status_detect = True
#             print('Detect status: {}'.format(status_detect))

#         def stopTurret():
#             global shot_status
#             shot_status = False
#             print('Turret Status: {}'.format(shot_status))

#         scheduler = BackgroundScheduler(timezone=get_localzone())
#         scheduler.add_job(task_deley, 'interval', seconds=sec_per_frame)
#         # scheduler.add_job(stopTurret, 'interval', seconds=20)
#         scheduler.start()

#         while True:
#             _, frame = cap.read()
#             if _:
#                 cen_x = frame.shape[1]
#                 cen_y = frame.shape[0]
#                 if status_detect:
#                     status_detect = False
#                     print("loop running")

#                     r = retinanet.detect(frame)
#                     print('return box', r)

#         cap.release()

#     _a(es, cam_api)
#     global status
#     status = False
#     # return redirect(url_for('index'))
#     return make_response('detect off')


if __name__ == '__main__':
    import socket
    if socket.gethostname() == 'minibear-jetson':
        app.run(host='0.0.0.0')
    else:
        app.run(host='0.0.0.0', debug=True)
