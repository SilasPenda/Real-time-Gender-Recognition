from keras.models import load_model
from keras_preprocessing.image import img_to_array
import numpy as np
import cv2
import cvlib as cv
import pyttsx3

# Load the model
model = load_model('GR.h5')

engine = pyttsx3.init()

# Select webcam
webcam = cv2.VideoCapture(0)

# Create gender classes
classes = {
    0: 'female',
    1: 'male'
}

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Loop through frames
while webcam.isOpened():
    # Read the frames from the webcam
    status, frame = webcam.read()
    
    faces = face_cascade.detectMultiScale(image = frame, scaleFactor = 1.1, minNeighbors = 3)
    for x, y, w, h in faces:
        # Draw rectangle over face
        cv2.rectangle(img = frame, pt1 = (x, y), pt2 = (x + w, y + h), color = (0, 255, 0), thickness = 2)
        
        # Do preprocessing based on model
        face_crop = frame[y:y + h, x:x + w]
        face_crop = cv2.resize(face_crop, (224, 224))
        face_crop = img_to_array(face_crop)
        face_crop = face_crop / 255
        face_crop = np.expand_dims(face_crop, axis = 0)
        
        # Predict gender
        prediction = model.predict(face_crop)[0]
        
        # Get the max accuracy
        idx = prediction.argmax(axis=-1)
        
        # Get the label using the max accuracy
        label_class = classes[idx]
        
        # Create the format for label and confidence (%) to be displayed
        label = '{}: {:2f}%'.format(label_class, prediction[idx] * 100)
        
        # # Write label and confidence above the face rectangle
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the output    
    cv2.imshow('Gender Detection', frame)
    
    # Press 's' to stop
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()