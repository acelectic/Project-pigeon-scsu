import os, cv2
from flask import Flask, render_template, Response, request, redirect, url_for
from multiprocessing import Process
from importlib import import_module


if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from until.camera_opencv import Camera

app = Flask(__name__)
status = False
confidence = 0.5
detect_every_frame = 2

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

    elif subpath == 'confidence':
        confidence = data['confidence']


    return redirect(url_for('index'))

@app.route('/snap', methods=["GET"])
def snap(ip = '127.0.0.1'):
    print('from',ip)
    return render_template('live.html', ip=ip)

def run(vdo_ = 0):
    from retinanet import retinanet_model
    retinanet = retinanet_model.Model(confidence=confidence)
    vdo_ = 'retinanet/video/YouTube5.mp4'
    # cap = cv2.VideoCapture(vdo_)
    cap = cv2.VideoCapture(vdo_)
    frameindex = 1
    while True:
        _, frame = cap.read()
        if _  and frameindex % int(detect_every_frame) == 0:
            retinanet.detect(frame)
            print("loop running")

        frameindex += 1
    cap.release()

    global status
    status = False
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(host='0.0.0.0')
