import cv2

from until.base_camera import BaseCamera


class Camera(BaseCamera):
    video_source = 0
    camera = None

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        global camera
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()

    @staticmethod
    def reset_camera():
        camera.release()
