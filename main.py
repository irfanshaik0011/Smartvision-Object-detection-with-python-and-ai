#  Importing the libraries
#numpy is used for handling arrays (e.g., image manipulations).
#cv2 is the OpenCV library, which is widely used for computer vision tasks. The cv2.dnn module is used here to handle deep neural network operations.

import numpy as np
import cv2
classNames = {0: 'background',
              1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
              5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
              10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
              14: 'motorbike', 15: 'person', 16: 'pottedplant',
              17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

#creating the Model Files
#proto refers to the prototxt file, which is the model architecture (a description of the neural network).
#weights refers to the caffemodel file, which contains the trained weights for the MobileNet SSD model.
#These are the learned parameters that allow the model to make predictions.

proto = "MobileNetSSD_deploy.prototxt"
weights = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(proto, weights)

img = cv2.imread("img_2.png")
img_resized = cv2.resize(img, (300, 300))

blob = cv2.dnn.blobFromImage(img_resized, 0.007843, (300, 300),
                             (127.5, 127.5, 127.5), False)
net.setInput(blob)

detections = net.forward()

final = detections.squeeze()

height, width, _ = img.shape

font = cv2.FONT_HERSHEY_SIMPLEX

ig = img.copy()
for i in range(final.shape[0]):
    conf = final[i, 2]
    if conf > 0.5:
        class_name = classNames[final[i, 1]]
        x1n, y1n, x2n, y2n = final[i, 3:]
        x1 = int(x1n * width)
        y1 = int(y1n * height)
        x2 = int(x2n * width)
        y2 = int(y2n * height)
        top_left = (x1, y1)
        bottom_right = (x2, y2)
        ig = cv2.rectangle(ig, top_left, bottom_right, (0, 255, 0), 3)
        ig = cv2.putText(ig, class_name, (x1, y1 - 10), font, 0.5,
                         (255, 0, 0), 1, cv2.LINE_AA)


cv2.imshow('Detected Objects', ig)
cv2.waitKey(0)
cv2.destroyAllWindows()
