import asyncio
from http import client
import websockets
import cv2
import threading
import requests

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from imutils.video import VideoStream
import imutils
import time
import os

def background_task(interval_sec):
    # run forever
    global log
    global url
    print("masuk")
    while True:
        print("inserting to db")
        # perform the task
        r = requests.post(url, json={'detail':log})
        # print(r.text)
        print("Background: " + str(log))
        # block for the interval
        time.sleep(interval_sec)

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

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = model.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)


async def server(websocket, path):
    connected.add(websocket)
    print(connected)
    print("someone just connected")

    global log
    # background_thread(websocket)
    
    while True:
        try:
            while(vid.isOpened()):
                _, frame = vid.read()
                #scale down image
                # frame = cv2.imread('test6.png')
                frame = cv2.resize(frame, (711, 400)) 
                # frame = cv2.resize(frame, (640,480)) 
                
                count = [0,0,0]
                (locs, preds) = detect_and_predict_mask(frame, net, model, thresh_confidence)
                
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
                        count[0] += 1
                    elif (NoFacemask >= Incorrect and NoFacemask >= Correct):
                        label = "No Face Mask"
                        color = (0, 0, 255)
                        count[2] += 1
                    else:
                        label = "Incorrect"
                        color = (255, 0, 0)
                        count[1] += 1

                    # include the probability in the label
                    # label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                    label = "{}: {:.1f}%".format(label, max(Correct, Incorrect, NoFacemask) * 100)

                    # display the label and bounding box rectangle on the output
                    # frame
                    cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]
                man = cv2.imencode('.jpg', frame, encode_param)[1]
                result = "{}/{}/{}".format(count[0], count[1], count[2])
                log = f'{{"correct":"{count[0]}","incorrect":"{count[1]}", "no_mask":"{count[2]}"}}'
                # print(log)
                
                #appen man
                # man = man + "||{:02d}".format(2)
                
                #sender(man)
                await websocket.send(result)
                await websocket.send(man.tobytes())
                
        except websockets.exceptions.ConnectionClosed as e:
            print("a client disconnected")
            print(e)
        finally:
            connected.remove(websocket)
            print(str(websocket) + "\nremoved")

#main program

thresh_confidence = 0.7

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
model = load_model("mask_detector_15.model")

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")

vid = cv2.VideoCapture(1)

print("[INFO] starting server...")
connected = set()

#start_server = websockets.serve(time, "192.168.84.117", 9997)  #ip jetson di hotspot hp
#start_server = websockets.serve(time, "192.168.0.112", 9997)  #ip jetson wifi if
# start_server = websockets.serve(time, "192.168.0.135", 9997)  #ip local  
start_server = websockets.serve(server, "127.0.0.1", 9997)  #ip local  

log = None
url = "http://192.168.0.104:8000/api/add"
daemon = threading.Thread(target=background_task, args=(1,), daemon=True, name='Background')
daemon.start()

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

cv2.destroyAllWindows()
vid.stop()
