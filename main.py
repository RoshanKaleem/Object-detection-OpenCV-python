import numpy as np
import cv2 as cv

weightsPath = "frozen_inference_graph.pb"
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
# net = cv.dnn_DetectionModel(weightsPath, configPath)
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
# # (h, w) = img.shape[:2]
# # blob = cv.dnn.blobFromImage(cv.resize(img, (300, 300)), 0.007843,
# #                             (300, 300), 127.5)
# print("[INFO] computing object detections...")
# net.setInput(blob)
# detections = net.detect()



# for i in np.arange(0, detections.shape[2]):
#
#     confidences = detections[0, 0, i, 2]
#
#     if confidences > confidence:
#
#         idx = int(detections[0, 0, i, 1])
#         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#         (startX, startY, endX, endY) = box.astype("int")
#         # display the prediction
#         label = "{}: {:.2f}%".format(classNames[idx], confidence * 100)
#         print("[INFO] {}".format(label))
#         cv.rectangle(img, (startX, startY), (endX, endY),
#             COLORS[idx], 2)
#         y = startY - 15 if startY - 15 > 15 else startY + 15
#         cv.putText(img, label, (startX, y),
#             cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

# img = cv.imread('dog.jpeg')
# rows = img.shape[0]
# cols = img.shape[1]
# cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
# cvOut = cvNet.forward()
#
# for detection in cvOut[0,0,:,:]:
#     score = float(detection[2])
#     if score > 0.3:
#         left = detection[3] * cols
#         top = detection[4] * rows
#         right = detection[5] * cols
#         bottom = detection[6] * rows
#         cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
#
# cv.imshow('img', img)
#
# cv.imshow("Output", img)
cv.waitKey(0)

