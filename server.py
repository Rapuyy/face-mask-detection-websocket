
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import asyncio
import websockets
import cv2
import threading
import requests
import numpy as np
import time
import os
import argparse

def background_task(interval_sec):
    # run forever
    global log
    print("masuk")
    while True:
        if(log[0] == 0 and log[1] == 0 and log[2] == 0):
            continue
        print(log)
        print("inserting to db")
        r = requests.post(url, json={'detail':f'{{"correct":"{log[0]}","incorrect":"{log[1]}", "no_mask":"{log[2]}"}}'})
        time.sleep(interval_sec)

def detect_and_predict_mask(frame, net, model, thresh):
	(h, w) = frame.shape[:2]

	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()

	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > thresh:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]

			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				faces.append(face)
				locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = model.predict(faces, batch_size=32)

	return (locs, preds)


async def server(websocket, path):
    connected.add(websocket)
    print(websocket)
    global log

    while True:
        print("masuk loop")
        try:
            while(vid.isOpened()):
                _, frame = vid.read()
                frame = cv2.resize(frame, (960, 540)) 
                count = [0,0,0]

                (locs, preds) = detect_and_predict_mask(frame, net, model, thresh_confidence)
                
                for (box, pred) in zip(locs, preds):
                    (startX, startY, endX, endY) = box

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

                    label = "{}: {:.1f}%".format(label, max(Correct, Incorrect, NoFacemask) * 100)

                    cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]
                man = cv2.imencode('.jpg', frame, encode_param)[1]
                result = "{}/{}/{}".format(count[0], count[1], count[2])
                # log = f'{{"correct":"{count[0]}","incorrect":"{count[1]}", "no_mask":"{count[2]}"}}'
                log = count

                await websocket.send(result)
                await websocket.send(man.tobytes())
                
        except websockets.exceptions.ConnectionClosed as e:
            print("a client disconnected")
            print(e)
        finally:
            connected.remove(websocket)
            print(str(websocket) + "\nremoved")

#main program
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", type=str, default="deploy.prototxt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-mf", "--modelFace", type=str, default="res10_300x300_ssd_iter_140000.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-mm", "--modelMask", type=str, default="mask_detector_15_0.model",
	help="path to face mask model")
ap.add_argument("-ip", "--jetson", type=str, default="192.168.149.117",
	help="ip jetson")
ap.add_argument("-l", "--laravel", type=str, default="192.168.149.125",
	help="ip laravel")
args = vars(ap.parse_args())


print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join(["face_detector", args["prototxt"]])
weightsPath = os.path.sep.join(["face_detector", args["modelFace"]])
net = cv2.dnn.readNet(prototxtPath, weightsPath)
thresh_confidence = args["confidence"]

print("[INFO] loading face mask detector model...")
model = load_model(args["modelMask"])

print("[INFO] starting video stream...")
vid = cv2.VideoCapture(-1)

print("[INFO] starting server...")
connected = set()
start_server = websockets.serve(server, args["jetson"], 9997)  #ip local  

log = [0,0,0]
count = None
url = f'http://{args["laravel"]}:8000/api/add'
daemon = threading.Thread(target=background_task, args=(1,), daemon=True, name='Background')
daemon.start()

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()


cv2.destroyAllWindows()
vid.stop()
