# import the necessary packages
from objecttracker.centroidtracker import CentroidTracker
from objecttracker.trackableobject import TrackableObject
from imutils.video import FPS
from threading import Thread
import smtplib
import ssl
import numpy as np
import imutils
import dlib
import cv2
import datetime
from datetime import timedelta

# Variables that can be set according to programmer need
# Some thresholding values to be set here as well
interval = timedelta(seconds=60)
startTime = datetime.datetime.now()
endTime = startTime + interval
# For Email purpose
countOfEntered = 0
countOfExited = 0
skip_frames = 10
confidence_value = 0.4
video_file_path = "videos/4.mov"
output_file_path = "output/newFour.avi"
# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalPersonsEntered = 0
totalPersonsExited = 0


# Email Sending Function
def sendEmail(start, end, in0, out0):
    port = 465  # For SSL
    password = "Password212"
    sender_email = 'peoplecounter.saudfin@gmail.com'
    receiver_email = ['peoplecounter.saudfin@gmail.com']
    subject = 'People Counting System'
    s_time = start.strftime('%m/%d/%Y %H:%M:%S')
    e_time = end.strftime('%m/%d/%Y %H:%M:%S')
    body = """
    Number of people who exited and entered between %s and %s  are %d and %d.
    """ % (s_time, e_time, in0, out0)
    # Create a secure SSL context
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login(sender_email, password)
    email_text = """\ 
    From: %s  
    To: %s  
    Subject: %s
    
    %s
    """ % (sender_email, ", ".join(receiver_email), subject, body)

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', port)
        server.ehlo()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, email_text)
        server.close()
        print('Email successfully sent to ', receiver_email, '!')
        startTime = datetime.datetime.now()
    except Exception:
        print('Something went wrong...')


# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("mobilenet_ssd/MobileNetSSD_deploy.prototxt",
                               "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")

# if a video path was not supplied, grab a reference to the webcam
# if not args.get("input", False):
#     print("[INFO] starting video stream...")
#     vs = VideoStream(src=0).start()
#     time.sleep(2.0)

# otherwise, grab a reference to the video file
# else:
print("[INFO] opening video file...")
vs = cv2.VideoCapture(video_file_path)

# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a Trackable Object
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# start the frames per second throughput estimator
fps = FPS().start()

# Start Time
startTime = datetime.datetime.now()

# loop over frames from the video stream
while True:
    # grab the next frame and handle if we are reading from either
    # VideoCapture or VideoStream
    frame = vs.read()
    frame = frame[1]

    # if we are viewing a video and we did not grab a frame then we
    # have reached the end of the video
    if frame is None:
        break

    # resize the frame to have a maximum width of 500 pixels (the
    # less data we have, the faster we can process it), then convert
    # the frame from BGR to RGB for dlib
    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # if we are supposed to be writing a video to disk, initialize
    # the writer
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_file_path, fourcc, 30,
                                 (W, H), True)

    # initialize the current status along with our list of bounding
    # box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status = "Waiting"
    rects = []

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % skip_frames == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []

        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by requiring a minimum
            # confidence
            if confidence > confidence_value:
                # extract the index of the class label from the
                # detections list
                idx = int(detections[0, 0, i, 1])

                # if the class label is not a person, ignore it
                if CLASSES[idx] != "person":
                    continue

                # compute the (x, y)-coordinates of the bounding box
                # for the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                trackers.append(tracker)

    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    else:
        # loop over the trackers
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving 'up' or 'down'
    cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object
                if direction < 0 and centroid[1] < H // 2:
                    totalPersonsExited += 1
                    countOfExited += 1
                    to.counted = True

                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and centroid[1] > H // 2:
                    totalPersonsEntered += 1
                    countOfEntered += 1
                    to.counted = True

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("Exited Persons Count", totalPersonsExited),
        ("Entered Persons Count", totalPersonsEntered),
        ("Status", status),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # check to see if we should write the frame to disk
    if writer is not None:
        writer.write(frame)

    # show the output frame
    cv2.imshow("People Counter", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1
    fps.update()
    newTime = datetime.datetime.now()
    # Send Email after some interval
    if newTime >= endTime:
        thread = Thread(target=sendEmail, args=(startTime, endTime, countOfEntered, countOfExited))
        thread.start()
        startTime = datetime.datetime.now()
        endTime = startTime + interval
        countOfEntered = countOfExited = 0

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()

# if we are not using a video file, stop the camera video stream
# if not args.get("input", False):
#     vs.stop()

# otherwise, release the video file pointer
# else:
vs.release()

print("Total People Entered:", totalPersonsEntered)
print("Total People Exited:", totalPersonsExited)

# close any open windows
cv2.destroyAllWindows()
