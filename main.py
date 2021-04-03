import numpy as np
import cv2 as cv

weightsPath = "frozen_inference_graph.pb"
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

confidence = 0.7
classNames = []
with open('coco.names', 'r') as f:
    classNames = f.read().splitlines()
print(classNames)

# COLORS = np.random.uniform(0, 255, size=(len(classNames), 3))

net = cv.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean(127.5)
net.setInputSwapRB(True)
img = cv.imread('5200.jpg')
img = cv.resize(img, (500, 500))

classIds, confs, bbox = net.detect(img, confThreshold=0.5)
print(classIds,bbox)

for classIds, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
    cv.rectangle(img, box, color=(0,255,0),thickness=2)
    cv.putText(img,classNames[classIds-1], (box[0]+10, box[1]+30), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))

cv.imshow("image", img)

cv.waitKey(0)

