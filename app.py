import os, time, cv2
from flask import Flask, render_template, Response, jsonify, request, make_response, redirect, url_for
from multiprocessing import Process, Value, Pool
from importlib import import_module


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
porbRate = .8
detect_every_frame = 2

# vdo4test = glob.glob('video/*.mp4')
# def get_my_ip():
#     return jsonify({'ip': request.remote_addr}), 200

@app.route('/', methods=['POST', 'GET'])
def index():
    """Video streaming home page."""
    if request.method == 'POST':
        data = request.form
        if data['snap'] == "go2snap":
            return redirect(url_for('snap', ip=request.remote_addr))


    print(request.remote_addr)
    return render_template('index.html', status = 'on' if status else 'off', _pass='pigeon' , porbRate= porbRate ,detect_every_frame= detect_every_frame)


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

# @app.route('/mode/<path:subpath>', methods=["POST"])
# def mode(subpath):
#     import time
#     global p, status
#     # print(request.subject)
#     try :
#         data = request.get_json(force=True)
#     except:
#         data = request.form
#     print(data)
#     if data['pass'] == 'pigeon':
#         # run()
#         # print('pass is correct')
#         if subpath == 'off':
#             if status:
#                 p.terminate()
#                 print('Detect ON')
#                 status = False
#                 # return "Detect OFF"
#             # else:
#                 # return 'Detect is OFF'
#
#
#         elif subpath == 'on':
#             if status:
#                 # return 'Detect is On'
#                 pass
#             else:
#                 status = True
#                 p = Process(target=run, args=())
#                 p.start()
#                 # return 'Detect ON'
#
#     else:
#         return 'invalid'
#
#     return redirect(url_for('index'))

# @app.route('/set/<path:subpath>', methods=["POST"])
# def set(subpath):
#     import time
#     global detect_every_frame, porbRate
#     # print(request.form['frame'])
#     try :
#         data = request.get_json(force=True)
#     except:
#         data = request.form
#     if subpath == 'frame':
#         detect_every_frame = data['frame']
#     elif subpath == 'Confidence':
#         confidence = data['Confidence']
#         yolo.setConfidence(confidence)
#     return redirect(url_for('index'))

@app.route('/snap', methods=["GET"])
def snap(ip = '127.0.0.1'):
    print('from',ip)
    return render_template('live.html', ip=ip)

# def run():
#     print('run detect')
#     cap = cv2.VideoCapture(0)
#     frameindex = 1
#     while True:
#
#         _, frame = cap.read()
#         # print(frame)
#         # cv2.imshow('sss', frame)
#
#         # cv2.imwrite()
#         # cv2.waitKey()
#         out = yolo.detect_index(frame, frameindex)
#         frameindex += 1
#         cv2.imshow('ss', out)
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#         print("loop running")
#     cap.release()

# def run():
#     from retinanet import retinanet_model
#     retinanet = retinanet_model.Model()
#     print('run detect')
#     cap = cv2.VideoCapture(0)
#     frameindex = 1
#     while True:
#
#         _, frame = cap.read()
#
#         out = retinanet.detect(frame)
#         frameindex += 1
#
#         cv2.imshow('ss', out)
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#         print("loop running")
#     cap.release()

# def test():
#     cap = cv2.VideoCapture('/home/minibear3e/Desktop/Pigeon Detect/Yolo_rasp/Peter_186.jpg')
#
#     _, img = cap.read()
#     yolo.detect(img)

# def fulltest():
#     # cap = cv2.VideoCapture(vdo4test[0])
#     cap = cv2.VideoCapture(0)
#     res = 1
#     frameindex = 1
#     while res:
#         print(frameindex)
#         res, img = cap.read()
#
#         out = yolo.detect_index(img, frameindex)
#         frameindex += 1
#         # cv2.imshow('ss', out)
#         # if cv2.waitKey(25) & 0xFF == ord('q'):
#         #     break

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    # test()
    # fulltest()