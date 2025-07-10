from imutils import paths
import pickle
import cv2
import os
import numpy as np

def simple_face_encoding(face_image, num_features=128):
    face_image = cv2.resize(face_image, (100, 100))
    if len(face_image.shape) == 3:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_image.copy()
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten()
    hist = hist / (hist.sum() + 1e-7)
    if len(hist) >= num_features:
        encoding = hist[:num_features]
    else:
        encoding = np.pad(hist, (0, num_features - len(hist)), mode='constant')
    return encoding.tolist()

def face_locations_simple(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    face_locations = []
    for (x, y, w, h) in faces:
        face_locations.append((y, x + w, y + h, x))
    return face_locations

def face_encodings_simple(image, face_locations):
    encodings = []
    for (top, right, bottom, left) in face_locations:
        face_image = image[top:bottom, left:right]
        if face_image.size > 0:
            encoding = simple_face_encoding(face_image)
            encodings.append(encoding)
    return encodings

print("[INFO] start processing faces...")
imagePaths = list(paths.list_images("dataset"))
knownEncodings = []
knownNames = []
for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    if image is None:
        print(f"[WARNING] Could not load image: {imagePath}")
        continue
    boxes = face_locations_simple(image)
    encodings = face_encodings_simple(image, boxes)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
print(f"[INFO] Training completed! Found {len(knownEncodings)} face encodings for {len(set(knownNames))} people.")
print("[INFO] encodings.pickle file created successfully.")
