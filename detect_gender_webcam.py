from keras.models import load_model
from keras_preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
import os
import cv2 as cv

# Load the model
model = load_model('GR.h5')

# Select webcam
webcam = cv2.VideoCapture(0)

# Create gender classes
classes = {
    0: 'female',
    1: 'male'
}

# Loop through frames
while webcam.isOpened():
    # Read the frames from the webcam
    status, frame = webcam.read()
    
    # Apply face detection
    face, confidence = cv.detect_face(frame)
    
    # Loop through the detected faces
    for idx, f in enumerate(face):
        
        # Get corners of the face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3] 
        
        # Draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        
        # Crop the deted face
        face_crop = np.copy(frame[startY:endY, startX:endX])
        
        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue        
        
        # Do preprocessing based on model
        face_crop = cv2.resize(face_crop, (224, 224))
        face_crop = img_to_array(face_crop)
        face_crop = face_crop/255
        face_crop = np.expand_dims(face_crop, [0])
        
        # Predict gender
        conf = model.predict(face_crop)[0]
        
        # Get the max accuracy
        idx = conf.argmax(axis=-1)
        
        # Get the label using the max accuracy
        label = classes[idx]
        
        # Create the format for label and confidence (%) to be displayed
        label = '{}: {:2f}%'.format(label, conf[idx] * 100)
        
        Y = startY - 10 if startY -10 > 10 else startY + 10
        
        # Write label and confidence above the face rectangle
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    # Display the output    
    cv2.imshow('Gender Detection', frame)

    # Press 's' to stop
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()