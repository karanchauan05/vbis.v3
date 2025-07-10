#! /usr/bin/python

from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import pickle
import time
import cv2
import numpy as np
import os

def simple_face_encoding(face_image, num_features=128):
	face_image = cv2.resize(face_image, (100, 100))
	if len(face_image.shape) == 3:
		gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
	else:
		gray = face_image.copy()
	hist_full = cv2.calcHist([gray], [0], None, [64], [0, 256])
	hist_full = hist_full.flatten()
	hist_full = hist_full / (hist_full.sum() + 1e-7)
	upper_half = gray[:50, :]
	hist_upper = cv2.calcHist([upper_half], [0], None, [32], [0, 256])
	hist_upper = hist_upper.flatten()
	hist_upper = hist_upper / (hist_upper.sum() + 1e-7)
	lower_half = gray[50:, :]
	hist_lower = cv2.calcHist([lower_half], [0], None, [32], [0, 256])
	hist_lower = hist_lower.flatten()
	hist_lower = hist_lower / (hist_lower.sum() + 1e-7)
	encoding = np.concatenate([hist_full, hist_upper, hist_lower])
	if len(encoding) >= num_features:
		encoding = encoding[:num_features]
	else:
		encoding = np.pad(encoding, (0, num_features - len(encoding)), mode='constant')
	return encoding

def compare_faces_opencv(known_encodings, face_encoding, tolerance=0.15):
	if len(known_encodings) == 0:
		return []
	distances = []
	for known_encoding in known_encodings:
		distance = np.linalg.norm(np.array(known_encoding) - np.array(face_encoding))
		distances.append(distance)
	matches = [distance <= tolerance for distance in distances]
	return matches, distances

def face_locations_opencv(frame):
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	face_locations = []
	for (x, y, w, h) in faces:
		face_locations.append((y, x + w, y + h, x))
	return face_locations

def face_encodings_opencv(frame, face_locations):
	encodings = []
	for (top, right, bottom, left) in face_locations:
		face_image = frame[top:bottom, left:right]
		if face_image.size > 0:
			encoding = simple_face_encoding(face_image)
			encodings.append(encoding)
	return encodings

currentname = "unknown"
encodingsP = "encodings.pickle"

if not os.path.exists(encodingsP):
	print(f"[ERROR] {encodingsP} not found!")
	print("Please run train_model.py first to create the encodings file.")
	exit()

print("[INFO] loading encodings...")
data = pickle.loads(open(encodingsP, "rb").read())

print("[INFO] starting video stream...")
vs = VideoStream(src=0, framerate=10).start()
time.sleep(2.0)
fps = FPS().start()

while True:
	frame = vs.read()
	if frame is None:
		break
	frame = imutils.resize(frame, width=500)
	boxes = face_locations_opencv(frame)
	encodings = face_encodings_opencv(frame, boxes)
	names = []
	for encoding in encodings:
		matches, distances = compare_faces_opencv(data["encodings"], encoding)
		name = "Other"
		if True in matches:
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			if matchedIdxs:
				best_match_idx = min(matchedIdxs, key=lambda i: distances[i])
				best_distance = distances[best_match_idx]
				if best_distance < 0.1:
					name = data["names"][best_match_idx]
					if currentname != name:
						currentname = name
						print(f"Recognized: {currentname} (distance: {best_distance:.3f})")
				else:
					name = "Other"
					if currentname != "Other":
						currentname = "Other"
						print(f"Unknown person detected (best distance: {best_distance:.3f})")
			else:
				name = "Other"
		else:
			name = "Other"
			if distances:
				min_distance = min(distances)
				if currentname != "Other":
					currentname = "Other"
					print(f"Unknown person detected (closest distance: {min_distance:.3f})")
		names.append(name)
	for ((top, right, bottom, left), name) in zip(boxes, names):
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
				   .8, (0, 255, 255), 2)
	cv2.imshow("Facial Recognition is Running (OpenCV)", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()
