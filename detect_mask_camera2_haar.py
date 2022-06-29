# import the necessary packages

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import numpy as np

def detect_and_predict_mask(frame, detector, model):
	faces = []
	locs = []
	preds = []
	(h, w) = frame.shape[:2]
	
	detections = detector.detectMultiScale(gray, scaleFactor=1.05,
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

	for i in range(0, len(detections)):
		box = detections[i]
		(startX, startY, endX, endY) = box

		endX = startX + endX
		endY = startY + endY

		# (startX, startY) = (max(0, startX), max(0, startY))
		# (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

		face = frame[startY:endY, startX:endX]
		if face.any():
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = model.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", type=str,
	default="haarcascade_frontalface_default.xml",
	help="path to haar cascade face detector")
args = vars(ap.parse_args())

print("[INFO] loading face detector...")
detector = cv2.CascadeClassifier(args["cascade"])
# initialize the video stream and allow the camera sensor to warm up

print("[INFO] loading face mask detector model...")
model = load_model("mask_detector_15.model")

print("[INFO] starting video stream...")
vs = VideoStream(src=1).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	start_time = time.time()
	
	# grab the frame from the video stream, resize it, and convert it
	# to grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# perform face detection

	(locs, preds) = detect_and_predict_mask(frame, detector, model)
	# loop over the bounding boxes
	for (box, pred) in zip(locs, preds):
		# print(pred)
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		# (mask, withoutMask) = pred

		# # determine the class label and color we'll use to draw
		# # the bounding box and text
		# label = "Mask" if mask > withoutMask else "No Mask"
		# color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		(Correct, Incorrect, NoFacemask) = pred
		if (Correct >= Incorrect and Correct >= NoFacemask):
			label = "Correct"
			color = (0, 255, 0)
		elif (NoFacemask >= Incorrect and NoFacemask >= Correct):
			label = "No Face Mask"
			color = (0, 0, 255)
		else:
			label = "Incorrect"
			color = (255, 0, 0)

		# include the probability in the label
		# label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		label = "{}: {:.1f}%".format(label, max(Correct, Incorrect, NoFacemask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
	# show the output frame
	cv2.imshow("Frame", frame)

	#print fps in terminal
	print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop

	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()