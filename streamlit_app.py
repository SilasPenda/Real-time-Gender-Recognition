from io import BytesIO
import streamlit as st
import numpy as np
from PIL import Image, ImageColor
from keras.models import load_model
from keras_preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2

model = load_model('GR.h5')

# Create gender classes
classes = {
    0: 'female',
    1: 'male'
}

RTC_CONFIGURATION = RTCConfiguration(
{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Set page configs. Get emoji names from WebFx
st.set_page_config(page_title="Real-time Gender Detection", page_icon="./assets/faceman_cropped.png", layout="centered")

# -------------Header Section------------------------------------------------

title = '<p style="text-align: center;font-size: 40px;font-weight: 550; "> Real-time Gender Detection</p>'
st.markdown(title, unsafe_allow_html=True)

st.markdown(
    "Gender Recognition using *Haar-Cascade Algorithm*, *OpenCV*, *and* *Tensorflow*.")

supported_modes = "<html> " \
                  "<body><div> <b>Supported Face Detection Modes (Change modes from sidebar menu)</b>" \
                  "<ul><li>Image Upload</li><li>Webcam Image Capture</li><li>Webcam Video Realtime</li></ul>" \
                  "</div></body></html>"
st.markdown(supported_modes, unsafe_allow_html=True)

st.warning("NOTE : Click the arrow icon at Top-Left to open Sidebar menu. ")

# -------------Sidebar Section------------------------------------------------

detection_mode = None
# Haar-Cascade Parameters
minimum_neighbors = 4
# Minimum possible object size
min_object_size = (50, 50)
# bounding box thickness
bbox_thickness = 3
# bounding box color
bbox_color = (0, 255, 0)

with st.sidebar:
    st.image("./assets/faceman_cropped.png", width=260)

    title = '<p style="font-size: 25px;font-weight: 550;">Face Detection Settings</p>'
    st.markdown(title, unsafe_allow_html=True)

    # choose the mode for detection
    mode = st.radio("Choose Face Detection Mode", ('Image Upload',
                                                   'Webcam Image Capture',
                                                   'Webcam Real-time'), index=0)
    if mode == 'Image Upload':
        detection_mode = mode
    elif mode == "Webcam Image Capture":
        detection_mode = mode
    elif mode == 'Webcam Real-time':
        detection_mode = mode

    # slider for choosing parameter values
    minimum_neighbors = st.slider("Mininum Neighbors", min_value=0, max_value=10,
                                  help="Parameter specifying how many neighbors each candidate "
                                       "rectangle should have to retain it. This parameter will affect "
                                       "the quality of the detected faces. Higher value results in less "
                                       "detections but with higher quality.",
                                  value=minimum_neighbors)

    # slider for choosing parameter values

    min_size = st.slider(f"Mininum Object Size, Eg-{min_object_size} pixels ", min_value=3, max_value=500,
                         help="Minimum possible object size. Objects smaller than that are ignored.",
                         value=70)

    min_object_size = (min_size, min_size)

    # Get bbox color and convert from hex to rgb
    bbox_color = ImageColor.getcolor(str(st.color_picker(label="Bounding Box Color", value="#00FF00")), "RGB")

    # ste bbox thickness
    bbox_thickness = st.slider("Bounding Box Thickness", min_value=1, max_value=30,
                               help="Sets the thickness of bounding boxes",
                               value=bbox_thickness)

    st.info("NOTE : The quality of detection will depend on above paramters."
            " Try adjusting them as needed to get the most optimal output")

    # line break
    st.markdown(" ")

    # About the programmer
    st.markdown("## Made by *Penda Silas, Ogunjimi Ayobami, Ebenezer Acquah, Olabisi Oluwale Anthony, Raphael Okai, and Oluwatimilehin Folarin* \U0001F609")
    st.markdown("[*Github Repo*](https://github.com/SilasPenda/Real-time-Gender-Detection)")

# -------------Image Upload Section------------------------------------------------


if detection_mode == "Image Upload":
    # Example Images
    col1, col2 = st.columns(2)
    with col1:
        st.image(image="./assets/example_2.png")
    with col2:
        st.image(image="./assets/example_3.png")

    uploaded_file = st.file_uploader("Upload Image (Only PNG & JPG images allowed)", type=['png', 'jpg'])

    if uploaded_file is not None:

        with st.spinner("Detecting faces..."):
            img = Image.open(uploaded_file)

            # To convert PIL Image to numpy array:
            img = np.array(img)

            # Load the cascade
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

            # Convert into grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray_img, 1.1, minNeighbors=minimum_neighbors, minSize=min_object_size)

            if len(faces) == 0:
                st.warning(
                    "No Face Detected in Image. Make sure your face is visible in the camera with proper lighting."
                    " Also try adjusting detection parameters")
            else:
                # Draw rectangle around the faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=bbox_color, thickness=bbox_thickness)
                    # Do preprocessing based on model
                    face_crop = img[y:y + h, x:x + w]
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
                    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the output
                st.image(img)

                if len(faces) > 1:
                    st.success("Total of " + str(
                        len(faces)) + " faces detected inside the image. Try adjusting minimum object size if we missed anything")

                    # convert to pillow image
                    img = Image.fromarray(img)
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")

                    # Creating columns to center button
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        pass
                    with col3:
                        pass
                    with col2:
                        st.download_button(
                            label="Download image",
                            data=buffered.getvalue(),
                            file_name="output.png",
                            mime="image/png")
                else:
                    st.success(
                        "Only 1 face detected inside the image. Try adjusting minimum object size if we missed anything.")

                    # convert to pillow image
                    img = Image.fromarray(img)
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")

                    # Creating columns to center button
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        pass
                    with col3:
                        pass
                    with col2:
                        st.download_button(
                            label="Download image",
                            data=buffered.getvalue(),
                            file_name="output.png",
                            mime="image/png")

# -------------Webcam Image Capture Section------------------------------------------------

if detection_mode == "Webcam Image Capture":

    st.info("NOTE : In order to use this mode, you need to give webcam access.")

    img_file_buffer = st.camera_input("Capture an Image from Webcam", disabled=False, key=1,
                                      help="Make sure you have given webcam permission to the site")

    if img_file_buffer is not None:

        with st.spinner("Detecting faces ..."):
            # To read image file buffer as a PIL Image:
            img = Image.open(img_file_buffer)

            # To convert PIL Image to numpy array:
            img = np.array(img)

            # Load the cascade
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

            # Convert into grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, minNeighbors=minimum_neighbors, minSize=min_object_size)

            if len(faces) == 0:
                st.warning(
                    "No Face Detected in Image. Make sure your face is visible in the camera with proper lighting. "
                    "Also try adjusting detection parameters")
            else:
                # Draw rectangle around the faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), color=bbox_color, thickness=bbox_thickness)
                    
                    # Do preprocessing based on model
                    face_crop = img[y:y + h, x:x + w]
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
                    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the output
                st.image(img)

                if len(faces) > 1:
                    st.success("Total of " + str(
                        len(faces)) + " faces detected inside the image. Try adjusting minimum object size if we missed anything")
                else:
                    st.success(
                        "Only 1 face detected inside the image. Try adjusting minimum object size if we missed anything")

                # Download the image
                img = Image.fromarray(img)
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                # Creating columns to center button
                col1, col2, col3 = st.columns(3)
                with col1:
                    pass
                with col3:
                    pass
                with col2:
                    st.download_button(
                        label="Download image",
                        data=buffered.getvalue(),
                        file_name="output.png",
                        mime="image/png")

# -------------Webcam Real-time Section------------------------------------------------


if detection_mode == "Webcam Real-time":

    # load face detection model
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    st.warning("NOTE : In order to use this mode, you need to give webcam access. "
               "After clicking 'Start' , it takes about 10-20 seconds to ready the webcam.")

    spinner_message = "Wait a sec, getting some things done..."

    with st.spinner(spinner_message):
        class VideoProcessor:
            def recv(self, frame):
                img = frame.to_ndarray(format = 'bgr24')
                faces = face_cascade.detectMultiScale(image=img, scaleFactor=1.1, minNeighbors=minimum_neighbors, minSize=min_object_size)

                for (x, y, w, h) in faces:
                    # Draw rectangle over face
                    cv2.rectangle(img = img, pt1 = (x, y), pt2 = (x + w, y + h), color = (0, 255, 0), thickness = 2)
                    
                    # Do preprocessing based on model
                    face_crop = img[y:y + h, x:x + w]
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
                    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                return av.VideoFrame.from_ndarray(img, format = 'bgr24')


        webrtc_streamer(key = 'example',
                        rtc_configuration = RTC_CONFIGURATION,
                        video_processor_factory = VideoProcessor,
                        media_stream_constraints = {
                            'video': True,
                            'audio': False
                            }
                        )

# -------------Hide Streamlit Watermark------------------------------------------------
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)