from PIL import Image, ImageDraw
import face_recognition
import os

def eye_cropper(folders):
    # Establish count for iterative file saving
    count = 0

    # For loop going through each image file
    for folder in os.listdir(folders):
        for file in os.listdir(folders + '/' + folder):
          
            # Using Facial Recognition Library on Image
            image = face_recognition.load_image_file(folders + '/' + folder + '/' + file)
            # create a variable for the facial feature coordinates
            face_landmarks_list = face_recognition.face_landmarks(image)
          
            # create a placeholder list for the eye coordinates
            eyes = []
            try:
                eyes.append(face_landmarks_list[0]['left_eye'])
                eyes.append(face_landmarks_list[0]['right_eye'])
            except:
                continue
            # establish the max x and y coordinates of the eye
            for eye in eyes:
                x_max = max([coordinate[0] for coordinate in eye])
                x_min = min([coordinate[0] for coordinate in eye])
                y_max = max([coordinate[1] for coordinate in eye])
                y_min = min([coordinate[1] for coordinate in eye])
              # establish the range of x and y coordinates    
                x_range = x_max - x_min
                y_range = y_max - y_min
              
                # to make sure the full eye is captured,
                # calculate the coordinates of a square that has 50%
                # cushion added to the axis with a larger range
                if x_range > y_range:
                    right = round(.5*x_range) + x_max
                    left = x_min - round(.5*x_range)
                    bottom = round(((right-left) - y_range))/2 + y_max
                    top = y_min - round(((right-left) - y_range))/2
                else:
                    bottom = round(.5*y_range) + y_max
                    top = y_min - round(.5*y_range)
                    right = round(((bottom-top) - x_range))/2 + x_max
                    left = x_min - round(((bottom-top) - x_range))/2
              
                #crop original image using the cushioned coordinates
                im = Image.open(folders + '/' + folder + '/' + file)
                im = im.crop((left, top, right, bottom))
                
                # resize image for input into our model
                im = im.resize((80,80))
              
                # save file to output folder
                im.save('C:/Eyes/ClosedFaceEyes/{}.jpg'.format(count))
              
                # increase count for iterative file saving
                count += 1
                # print count every 200 photos to monitor progress
                if count % 200 == 0:
                    print(count)
    
# Call function to crop full-face eye images
eye_cropper('C:/Eyes/Dataset/ClosedFace')