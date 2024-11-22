import numpy as np
import cv2

# Define class names and load model
classNames = {0: 'background',
              1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
              5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
              10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
              14: 'motorbike', 15: 'person', 16: 'pottedplant',
              17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

proto = "MobileNetSSD_deploy.prototxt"
weights = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(proto, weights)

# Access live stream from IP camera
# Replace 'your_ip_address' with the actual IP address of the camera
# if you want your system default camera add 0 or 1 depends on your camera
camera_url = 0
#camera_url = "https://192.168.0.112:8080"

cap = cv2.VideoCapture(camera_url)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Unable to open camera stream")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # If frame reading is successful
    if ret:
        # Preprocess the frame
        frame_resized = cv2.resize(frame, (300, 300))
        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        net.setInput(blob)

        # Perform object detection
        detections = net.forward()
        final = detections.squeeze()

        height, width, _ = frame.shape
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw bounding boxes and labels on the frame
        for i in range(final.shape[0]):
            conf = final[i, 2]
            if conf > 0.5:
                class_name = classNames[int(final[i, 1])]
                x1n, y1n, x2n, y2n = final[i, 3:]
                x1 = int(x1n * width)
                y1 = int(y1n * height)
                x2 = int(x2n * width)
                y2 = int(y2n * height)
                top_left = (x1, y1)
                bottom_right = (x2, y2)
                frame = cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)
                frame = cv2.putText(frame, class_name, (x1, y1 - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Live Stream with Object Detection', frame)

        # Check for key press
        key = cv2.waitKey(1)

        # If 'q' key is pressed, exit the loop
        if key == ord('q'):
            break
    else:
        print("Error: Unable to read frame from camera stream")
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
