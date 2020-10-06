# import the necessary packages
import os
import time

import cv2
import numpy as np
from imutils.video import FPS

RTSP_URL = "rtsp://admin:@192.168.88.93:80/ch0_0.264"
YOLO_PATH = "yolo-coco"
OUTPUT_FILE = "output/outfile0.avi"

width = 640
height = 480

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([YOLO_PATH, "data", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
CONFIDENCE = 0.5
THRESHOLD = 0.3

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([YOLO_PATH, "yolov3.weights"])
configPath = os.path.sep.join([YOLO_PATH, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes) and determine only the *output* layer names that
# we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and frame dimensions
vs = cv2.VideoCapture(RTSP_URL)
videoWidth = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
videoHeight = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
time.sleep(2.0)
fps = FPS().start()
writer = None
(W, H) = (None, None)

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward pass of the YOLO object detector,
    # giving us our bounding boxes and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # draw area of detection
    cv2.rectangle(frame, (5, 5), (videoWidth, 400), (0, 0, 255), thickness=5)

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected probability is greater than the minimum probability
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

    # ensure at least one detection exists
    if len(idxs) > 0:
        px1, py1, px2, py2 = 5, 5, videoWidth, 400

        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            hx1, hy1, hx2, hy2 = x, y, x + w, y + h

            if hx1 >= px1 and hy1 >= py1 and hx2 <= px2 and hy2 <= py2:
                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # write to image
                # cv2.imwrite("test.png", frame)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, 30, (frame.shape[1], frame.shape[0]), True)

    # write the output frame to disk
    writer.write(frame)

    # show the output frame
    cv2.imshow("RTSP Object Detection", cv2.resize(frame, (640, 480)))
    key = cv2.waitKey(1) & 0xFF
    # print ("key", key)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()

print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
