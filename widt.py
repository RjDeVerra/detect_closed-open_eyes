from flask import Flask,render_template,Response
import cv2
import face_recognition
import os
from tensorflow import keras
from playsound import playsound
# import model saved above
eye_model = keras.models.load_model('openclose.h5')

app=Flask(__name__)
camera=cv2.VideoCapture(0)

def eye_cropper(frame):

    # create a variable for the facial feature coordinates
    facial_features_list = face_recognition.face_landmarks(frame)

    # create a placeholder list for the eye coordinates
    # and append coordinates for eyes to list unless eyes
    # weren't found by facial recognition
    try:
        eye = facial_features_list[0]['left_eye']
    except:
        try:
            eye = facial_features_list[0]['right_eye']
        except:
            return
    
    # establish the max x and y coordinates of the eye
    x_max = max([coordinate[0] for coordinate in eye])
    x_min = min([coordinate[0] for coordinate in eye])
    y_max = max([coordinate[1] for coordinate in eye])
    y_min = min([coordinate[1] for coordinate in eye])

    # establish the range of x and y coordinates
    x_range = x_max - x_min
    y_range = y_max - y_min

    # in order to make sure the full eye is captured,
    # calculate the coordinates of a square that has a
    # 50% cushion added to the axis with a larger range and
    # then match the smaller range to the cushioned larger range
    if x_range > y_range:
        right = round(.5*x_range) + x_max
        left = x_min - round(.5*x_range)
        bottom = round((((right-left) - y_range))/2) + y_max
        top = y_min - round((((right-left) - y_range))/2)
    else:
        bottom = round(.5*y_range) + y_max
        top = y_min - round(.5*y_range)
        right = round((((bottom-top) - x_range))/2) + x_max
        left = x_min - round((((bottom-top) - x_range))/2)

    # crop the image according to the coordinates determined above
    cropped = frame[top:(bottom + 1), left:(right + 1)]

    # resize the image
    cropped = cv2.resize(cropped, (80,80))
    image_for_prediction = cropped.reshape(-1, 80, 80, 3)

    return image_for_prediction

font = cv2.FONT_HERSHEY_SIMPLEX
  
org = (200, 80)
  
fontScale = 3
   
# Blue color in BGR
green = (0,255,0)
red = (0, 0, 255)

thickness = 3

def generate_frames():
    while True:
        success,frame=camera.read()

        image_for_prediction = eye_cropper(frame)

        try:
            image_for_prediction = image_for_prediction/255.0
        except:
            continue

        prediction = eye_model.predict(image_for_prediction)

        if prediction < 0.5:
            status = 'Open'

        else:
            status = 'Closed'

        if status == 'Open':
            ret,buffer=cv2.imencode('.jpg', cv2.putText(frame, "{}".format(status), org, font, 
                       fontScale, green, thickness, cv2.LINE_AA))
            frame=buffer.tobytes()
        else:
            ret,buffer=cv2.imencode('.jpg', cv2.putText(frame, "{}".format(status), org, font, 
                       fontScale, red, thickness, cv2.LINE_AA))
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)