#DETEKSI MASKER via VideoCapture

# CARA CEK DI CMD
# python detect_mask_image.py --image examples/example_01.png
# python detect_mask_image.py --image Test/155.jpg
# python detect_mask_image.py --image Data/Ada/cewek1.jpg
# python detect_mask_image.py --image Data/test1.jpg

# Impor Paket / Library yang diperlukan
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from imutils.video import VideoStream
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, net, model, thresh):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	net.setInput(blob)
	detections = net.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > thresh:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
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

	# only make a predictions if at least one face as detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = model.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)


thresh_confidence = 0.5

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)
print(len(net.getLayerNames()), net.getLayerNames())


# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
model = load_model("mask_detector_15.model")

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")

#vc = VideoStream(src=-1).start()
vc = cv2.VideoCapture(1)
time.sleep(2)

# loop over the frames from the video stream
while True:
	# fps counter
	start_time = time.time()
 
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	(_, frame) = vc.read()
	frame = cv2.resize(frame, (711, 400))
	#frame = vc.read()
	#frame = imutils.resize(frame, width=400)
 
	# frame = cv2.resize(frame, width=400, height = 400, interpolation = cv2.INTER_AREA)
	# frame = cv2.imread("test6.png", cv2.IMREAD_COLOR)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, net, model, thresh_confidence)
	# loop over the detected face locations and their corresponding
	# locations
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
vc.stop()
