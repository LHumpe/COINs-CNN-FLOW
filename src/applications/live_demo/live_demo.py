'''
  Level | Level for Humans | Level Description
 -------|------------------|------------------------------------
  0     | DEBUG            | [Default] Print all messages
  1     | INFO             | Filter out INFO messages
  2     | WARNING          | Filter out INFO & WARNING messages
  3     | ERROR            | Filter out all messages
'''
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
from threading import Thread
import tensorflow as tf
import numpy as np

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

modelpath = 'src/models/weights/final_model_weights.hdf5'
model = tf.keras.models.load_model(modelpath)


class VideoStream:
    def __init__(self, src=0, name="WebcamVideoStream"):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)  # + cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


vs = VideoStream(src=0).start()

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while (True):
    frame = vs.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    try:
        x, y, w, h = faces[0]
        face = frame[y:y + h, x:x + w]
        img = tf.cast(face, tf.float32)
        img = tf.keras.applications.resnet.preprocess_input(img)
        img = tf.image.per_image_standardization(img)
        img = tf.image.resize_with_pad(img, 224, 224)
        img = np.array(img)
        sample = list()
        sample.append(img)

        prediction = model.predict(np.array(sample))[0][0]
        flow = (prediction > 0.5).astype('int32')
        proba = (1 - prediction) * 2 if flow == 1 else (0.5 - prediction) * 2
        # print('Flow:', flow, 'Probability:', proba)
        # draw a red rectangle around detected objects
        cv2.rectangle(frame, (int(x), int(y + h)), (int(
            x + w), int(y)), (0, 0, 255), thickness=2)
        cv2.putText(frame, 'Flow: ' + str(flow),
                    (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255))
        cv2.putText(frame, 'Probability: ' + str(proba),
                    (x, y - 7), cv2.FONT_HERSHEY_COMPLEX_SMALL, .5, (0, 0, 255))
    except IndexError:
        face = None

    # Show the image with a rectagle surrounding the detected objects
    cv2.imshow('Image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cv2.destroyAllWindows()
vs.stop()
