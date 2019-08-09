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

from until.elas_api import elas_api

from until.camera_control import camera_control

cam_api = camera_control()

es_ip = '192.168.1.29'
es_port = 9200
es = elas_api(ip=es_ip)
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
def index():
    """Video streaming home page."""
    ip = request.remote_addr
    if request.method == 'POST':
        data = request.form
        print(data)
        try:
            if data['live'] == "live_camera":
                return redirect(url_for('live_camera', ip=ip))
        except:
            pass

    print(request.remote_addr)
    return render_template('index.html', status='on' if status else 'off', _pass='pigeon', confidence=confidence,
                           sec_per_frame=sec_per_frame, ip=ip, fullip="http://{}:5000/".format(ip), es_status=es_status,
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
            p.start()
            print('Detect On')

    return redirect(url_for('index'))

@app.route('/detectStatus', methods=["POST"])
def getDetectStatus():
    global status
    print('check status: {}'.format( 'on' if status else 'off'))
    return 'on' if status else 'off'


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
        print(sec_per_frame)

    elif subpath == 'confidence':
        confidence = data['confidence']

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

    return make_response("rotate camera "+ cmd)
    # return redirect(url_for('snap'))

def run(vdo_=0):
    global status_detect, shot_status

    print("{:#^20}{}{:#^20}\nconfidence:{}\nSec per frame{}".format('', 'Detect ON', '', confidence, sec_per_frame))


    # from retinanet import retinanet_model
    # from datetime import datetime
    # retinanet = retinanet_model.Model(confidence=confidence,  es=es, es_mode=True)
    #
    #
    #
    # status_detect = False
    # shot_status = False
    # # vdo_ = 'video/YouTube4.mp4'
    # # cap = cv2.VideoCapture(vdo_)
    # cap = cv2.VideoCapture(vdo_)
    #
    # def task_deley():
    #     global status_detect
    #     status_detect = True
    #     print('Detect status: {}'.format(status_detect))
    #
    # scheduler = BackgroundScheduler(timezone=get_localzone())
    # scheduler.add_job(task_deley, 'interval', seconds=sec_per_frame)
    # scheduler.start()
    #
    #
    # while True:
    #
    #     _, frame = cap.read()
    #     if _ :
    #         if status_detect:
    #             if not shot_status:
    #                 status_detect = False
    #                 print("loop running")
    #                 r = retinanet.detect(frame)
    #                 retinanet.create_Tracker(bbox=r, frame=frame)
    #                 shot_status = True
    #                 print(r)
    #         if shot_status:
    #             shot_tmp = retinanet._updateTracker(frame)
    #
    # cap.release()
    #
    # global status
    # status = False
    # return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(host='0.0.0.0')
    # app.run(host='0.0.0.0', debug=True)
