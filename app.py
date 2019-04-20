from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    import cv2, numpy as np, argparse, time, glob, os, sys, subprocess, pandas, random, math
#Define variables and load classifier
camnumber = 0
video_capture = cv2.VideoCapture()
facecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
fishface = cv2.face.FisherFaceRecognizer_create()
try:
    fishface.read("trained_emoclassifier.xml")
except:
    print("no xml found. Using --update will create one.")
parser = argparse.ArgumentParser(description="Options for the emotion-based music player") #Create parser object
parser.add_argument("--update", help="Call to grab new images and update the model accordingly", action="store_true") #Add --update argument
args = parser.parse_args() #Store any given arguments in an object

facedict = {}

emotions = ["angry", "happy", "sad"]


def crop_face(clahe_image, face):
    for (x, y, w, h) in face:
        faceslice = clahe_image[y:y+h, x:x+w]
        faceslice = cv2.resize(faceslice, (350, 350))
    facedict["face%s" %(len(facedict)+1)] = faceslice
    return faceslice

def check_folders(emotions):
    for x in emotions:
        if os.path.exists("dataset\\%s" %x):
            pass
        else:
            os.makedirs("dataset\\%s" %x)

def recognize_emotion():
    predictions = []
    confidence = []
    for x in facedict.keys():
        pred, conf = fishface.predict(facedict[x])
        cv2.imwrite("images\\%s.png" %x, facedict[x])
        predictions.append(pred)
        confidence.append(conf)
    recognized_emotion = emotions[max(set(predictions), key=predictions.count)]
    print("I think you're %s" %recognized_emotion)
    
def grab_webcamframe():
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #Create CLAHE object
        clahe_image = clahe.apply(gray) #Apply CLAHE to grayscale image from webcam
        face = facecascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=10, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in face: #Draw rectangle around detected faces
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) #draw it on "frame", (coordinates), (size), (RGB color), thickness 2
        cv2.imshow("Hold q to capture your emotion", frame) #Display frame
        if cv2.waitKey(1) & 0xFF == ord('q'): #imshow expects a termination definition to work correctly, here it is bound to key 'q'
            out = cv2.imwrite('image.png', frame)
            break
    return clahe_image
def detect_face():
    clahe_image = grab_webcamframe()
    face = facecascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(face) == 1:
        faceslice = crop_face(clahe_image, face)
        return faceslice
    else:
        print("no/multiple faces detected, passing over frame")
def run_detection():
    while len(facedict) != 1:
        detect_face()
    recognize_emotion()
video_capture.open(camnumber)
run_detection()

    return recognized_emotion
