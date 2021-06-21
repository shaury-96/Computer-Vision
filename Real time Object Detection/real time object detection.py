import numpy as np
import imutils
import cv2
import time

proTxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"
confThresh = 0.2

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# print(COLORS)

print("Model is getting loaded...")
net = cv2.dnn.readNetFromCaffe(proTxt, model)
print("Model loaded successfully")
print("Camera is ON")
vs = cv2.VideoCapture(0)
time.sleep(2)

while True:
    ret, frame = vs.read()
    frame = imutils.resize(frame, width=500)
    # frame = cv2.resize(frame, (0, 0), fx=1, fy=0.5)
    # print(frame.shape)
    (h, w) = frame.shape[:2]
    # print(h,w)
    imResizeBlob = cv2.resize(frame, (300,300))
    blob = cv2.dnn.blobFromImage(imResizeBlob,
        0.007843, (300,300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    # print(detections.shape)
    detShape = detections.shape[2]
    # print(detShape)

    for i in np.arange(0, detShape):
        confidence = detections[0, 0, i, 2]
        if confidence > confThresh:
            idx = int(detections[0, 0, i, 1])
            print("ClassID:", detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(CLASSES[idx], confidence*100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)

            if startY - 15 > 15:
                Y = startY -15
            else:
                startY + 15
            cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_PLAIN, 0.5, COLORS[idx], 2)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
vs.release()
cv2.destroyAllWindows()

