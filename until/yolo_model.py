import cv2
import numpy as np
import base64, random
from tempfile import NamedTemporaryFile
import radar
# try:
#     from .elas_api import elas_api
#
#     # eapi = elas_api(ip='192.168.1.41', port=9200)
#     eapi = elas_api(ip='127.0.0.1', port=9200)  # pls fill elasticsearch IP
# except Exception as e:
#     eapi = elas_api()
#     print("ERROR", e)

from datetime import datetime, timedelta
from until.elasApi import elas_api

# elas_api = elas_api()
elas_api = elas_api('192.168.1.11')


def gen_datetime():

    return radar.random_date(
    start = datetime(year=2018, month=8, day=24),
    stop = datetime(year=2019, month=2, day=13))



class yolo_model:
    # Initialize the parameters
    confThreshold = 0.7  # Confidence threshold
    nmsThreshold = 0.4  # Non-maximum suppression threshold

    syolo = 640
    inpWidth = syolo  # Width of network's input image
    inpHeight = syolo # Height of network's input image

    elas_index ='detect_ubuntu--'+str(syolo)
    elas_type = 'image'

    def setConfidence(self, confidence):
        self.confThreshold = confidence

    def __init__(self, modelWeights="yolov2-tiny.weights", modelConfiguration="yolov2-tiny-original.cfg",
                 classesFile="coco.names"):

        # Give the configuration and weight files for the model and load the network using them.
        # modelConfiguration = '/home/minibear3e/Desktop/4Raspberry/model/yolov2-tiny-ori.cfg'
        # modelWeights = '/home/minibear3e/Desktop/4Raspberry/model/yolov2-tiny.weights'
        # classesFile = '/home/minibear3e/Desktop/4Raspberry/model/coco.names'

        # modelConfiguration = 'model/yolov3-tiny.cfg'
        # modelWeights = 'model/yolov3-tiny.weights'
        # classesFile = 'model/coco.names'

        modelConfiguration = 'model/yolov2-tinys.cfg'
        modelWeights = 'model/yolov2-tiny_15000.weights'
        classesFile = 'model/obj.names'
        classes = None

        with open(classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        try:
            self.net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        except:
            print("Can't load model")

    def img2string(self, img):
        retval, buffer = cv2.imencode('.jpg', img)

        # Convert to base64 encoding and show start of data
        return str(base64.b64encode(buffer))

    # Get the names of the output layers
    def getOutputsNames(self):
        # Get the names of all the layers in the network
        layersNames = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, frame, outs, frameindex, timestamp):

        # Draw the predicted bounding box
        def drawPred(frame, classId, conf, left, top, right, bottom):
            # Draw a bounding box.
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            label = '%.2f' % conf

            # Get the label for the class name and its confidence
            if self.classes:
                assert (classId < len(self.classes))
                label = '%s: %s' % (self.classes[classId], label)

            # Display the label at the top of the bounding box
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            print(label)

        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)

        if len(indices) > 0:
            # cv2.imshow('detxt', frame)
            _h, _w, _c = (300, 300, 3)
            small = (_h, _w)

            img = {}
            # timestamp = datetime.utcnow()
            # print(timestamp)

            img['numDetect'] = len(indices)
            img['frameindex'] = frameindex
            tsm = gen_datetime()
            print(tsm)

            img['my_timestamp'] = tsm
            img['image_shape'] = (_h, _w)
            img['orginal_image'] = self.img2string(cv2.resize(frame, small))

            tmp_data = {}

            for i in indices:
                # print(i)
                i = i[0]
                box = boxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
                tmp_data['box_' + str(i)] = {'class': self.classes[classIds[i]], 'prob': confidences[i], 'x1': left,
                                             'y1': top,
                                             'x2': left + width, 'y2': top + height}


            jpg_as_text = self.img2string(cv2.resize(frame, small))

            img['mark_img'] = jpg_as_text
            img['box'] = tmp_data

            timedelta = datetime.utcnow() - timestamp
            print(timedelta)
            img['processTime(sec)'] = timedelta.total_seconds()

            # elas_api.putData(index=self.elas_index, data=img)
            # tempjpg = NamedTemporaryFile().name.split('/')[-1]+'.jpg'
            # print(tempjpg)
            # cv2.imwrite('result/'+str(frameindex)+'.jpg', frame)
            return img

    # def detect_image(self, imgdir):
    #     index = 0
    #     # cap = cv2.VideoCapture(imgdir)
    #     # cap = cv2.VideoCapture("test4_000000.jpg")
    #     # cap = cv2.VideoCapture(1)
    #     # cap = cv2.VideoCapture('pigeon-2.mp4')
    #     m = 1
    #
    #     while cv2.waitKey(1) < 0:
    #
    #         # get frame from the video
    #         # hasFrame, frame = cap.read()
    #         frame = cv2.imread(imgdir)
    #         print(frame.shape)
    #         inpHeight, inpWidth = frame.shape[0], frame.shape[1]
    #         midx = inpWidth // 2
    #         midy = inpHeight // 2
    #         print(midy, midx)
    #
    #         fixsize = (416, 416)
    #         frame = cv2.resize(frame, fixsize)
    #         frames = [frame]
    #         # frames = [frame[0:midy, 0:midx, :], frame[0:midy, midx:inpWidth, :], frame[midy:inpHeight, 0:midx, :], frame[midy:inpHeight, midx:inpWidth, :]]
    #
    #         for i in frames:
    #             i = cv2.resize(i, fixsize)
    #             blob = cv2.dnn.blobFromImage(i, 1 / 255, fixsize, (0, 0, 0), 1, crop=True)
    #             a = np.reshape(blob[0], (fixsize[0], fixsize[1], 3))
    #             # Sets the input to the network
    #             self.net.setInput(blob)
    #             print(blob.shape)
    #             cv2.imshow('blob', a)
    #             # Runs the forward pass to get output of the output layers
    #             outs = self.net.forward(self.getOutputsNames())
    #             # Remove the bounding boxes with low confidence
    #             check = self.postprocess(i, outs)
    #
    #             # Put efficiency information. The function getPerfProfile returns the
    #             # overall time for inference(t) and the timings for each of the layers(in layersTimes)
    #             t, _ = self.net.getPerfProfile()
    #             label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    #             index += 1
    #
    #             print("index:", index, label, "\n")
    #
    #             cv2.putText(i, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    #
    #             # Write the frame with the detection boxes
    #             cv2.imshow('ori', frame)
    #             # cv2.imshow("out", i)
    #             # if check['numDetect'] > 0:
    #             #     cv2.waitKey()
    #             cv2.waitKey()
    #     # cap.release()
    #     # cv2.destroyAllWindows()

    def detect(self, frame):
        index = 0

        # print(frame.shape)
        # inpHeight, inpWidth = frame.shape[0], frame.shape[1]
        # midx = inpWidth // 2
        # midy = inpHeight // 2
        # print(midy, midx)

        fixsize = (self.inpHeight, self.inpWidth)

        blob = cv2.dnn.blobFromImage(frame, 1 / 255, fixsize, (0, 0, 0), 1, crop=True)
        # a = np.reshape(blob[0], (fixsize[0], fixsize[1], 3))
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.getOutputsNames())
        # Remove the bounding boxes with low confidence
        result = self.postprocess(frame, outs)

        # Put efficiency information. The function getPerfProfile returns the
        # overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = self.net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        print(label)
        # print(result)
        # index += 1
        # print("index:", index, label, "\n")
        # cv2.imshow('sss', frame)
        # cv2.waitKey()

    def detect_index(self, frame, frameindex):

        # frame = cv2.resize(frame, (640, 480))
        # print(frame.shape)
        # inpHeight, inpWidth = frame.shape[0], frame.shape[1]
        # midx = inpWidth // 2
        # midy = inpHeight // 2
        # print(midy, midx)
        timestamp = datetime.utcnow()
        fixsize = (self.inpHeight, self.inpWidth)

        blob = cv2.dnn.blobFromImage(frame, 1 / 255, fixsize, (0, 0, 0), 1, crop=False)
        # a = np.reshape(blob[0], (fixsize[0], fixsize[1], 3))
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.getOutputsNames())
        # Remove the bounding boxes with low confidence
        result = self.postprocess(frame, outs, frameindex, timestamp)

        # Put efficiency information. The function getPerfProfile returns the
        # overall time for inference(t) and the timings for each of the layers(in layersTimes)
        # t, _ = self.net.getPerfProfile()
        # label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        # print(label)
        # print(result)
        # index += 1
        # print("index:", index, label, "\n")
        # cv2.imshow('sss', frame)
        return frame

    def test_normal(self):
        cap = cv2.VideoCapture('/home/minibear3e/Desktop/4Raspberry/video/YouTube4.mp4')
        m = 1
        index = 1
        while cv2.waitKey(1) < 0:
            # get frame from the video
            hasFrame, frame = cap.read()
            # frame = cv2.imread(imgdir)
            print(frame.shape)
            inpHeight, inpWidth = frame.shape[0], frame.shape[1]
            midx = inpWidth // 2
            midy = inpHeight // 2
            print(midy, midx)

            # fixsize = (416, 416)
            fixsize = (self.inpHeight, self.inpWidth)
            # frame = cv2.resize(frame, fixsize)
            frames = [frame]
            # frames = [frame[0:midy, 0:midx, :], frame[0:midy, midx:inpWidth, :], frame[midy:inpHeight, 0:midx, :], frame[midy:inpHeight, midx:inpWidth, :]]

            for i in frames:
                # i = cv2.resize(i, fixsize)
                blob = cv2.dnn.blobFromImage(i, 1 / 255, fixsize, (0, 0, 0), 1, crop=False)
                # a = np.reshape(blob[0], (fixsize[0], fixsize[1], 3))
                # Sets the input to the network
                self.net.setInput(blob)
                # print(blob.shape)
                # cv2.imshow('blob', a)
                # Runs the forward pass to get output of the output layers
                outs = self.net.forward(self.getOutputsNames())
                # Remove the bounding boxes with low confidence
                check = self.postprocess(i, outs, frameindex=index, timestamp=datetime.utcnow())

                # Put efficiency information. The function getPerfProfile returns the
                # overall time for inference(t) and the timings for each of the layers(in layersTimes)
                t, _ = self.net.getPerfProfile()
                label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
                index += 1

                print("index:", index, label, "\n")

                cv2.putText(i, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

                # Write the frame with the detection boxes
                cv2.imshow('ori', frame)
                # cv2.imshow("out", i)
                # if check['numDetect'] > 0:
                #     cv2.waitKey()
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


# yolo = yolo_model()
# yolo.test_normal()
# img = cv2.VideoCapture