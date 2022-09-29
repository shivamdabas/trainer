from flask import Flask, redirect, url_for, request
app = Flask(__name__)

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# to display images we need collab specific package 
from google.colab.patches import cv2_imshow


# Initialize mediapipe drawing class - to draw the landmarks points.
mp_drawing = mp.solutions.drawing_utils

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose

"""calculateAngle function will calculate the angle in degress between three keypoints

BodyPartDimensions function will return the dimensions of a keypoint or a body part from landmarks data structure
"""

def calculateAngle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) -\
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    # check cord sys area
    if angle > 180.0:
        angle = 360 - angle

    return angle

def BodyPartDimensions(landmarks, bodyPart):
    return [
        landmarks[mp_pose.PoseLandmark[bodyPart].value].x,
        landmarks[mp_pose.PoseLandmark[bodyPart].value].y,
        landmarks[mp_pose.PoseLandmark[bodyPart].value].visibility
    ]

"""Pushup function is counting the number of pushups done according to the average arm angle of both left and right arm where an arm angle in turn is the angle between elbow, shoulder and wrist

We are also printing the image with angle and keypoints using opencv puttext function
"""

def Pushup(landmarks, counter, status, image):
  l_elbow = BodyPartDimensions(landmarks, "LEFT_ELBOW")
  l_shoulder = BodyPartDimensions(landmarks, "LEFT_SHOULDER")
  l_wrist = BodyPartDimensions(landmarks, "LEFT_WRIST")
  left_arm_angle =  calculateAngle(l_shoulder, l_elbow, l_wrist)
  
  
  r_shoulder = BodyPartDimensions(landmarks, "RIGHT_SHOULDER")
  r_elbow = BodyPartDimensions(landmarks, "RIGHT_ELBOW")
  r_wrist = BodyPartDimensions(landmarks, "RIGHT_WRIST")
  right_arm_angle = calculateAngle(r_shoulder, r_elbow, r_wrist)
  avg_arm_angle = (left_arm_angle + right_arm_angle)//2

  # using opencv displaying angle value on image
  cv2.putText(image, "Avg Arm Angle: {}".format(avg_arm_angle), (15,20),
  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA )


    # Render detections
  mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
  mp_drawing.DrawingSpec(color=(240, 128, 128), thickness=3, circle_radius=2), 
  mp_drawing.DrawingSpec(color=(100, 149, 237), thickness=2, circle_radius=2) 
  )               
        
       
  cv2_imshow(image)       
            
  if status == 0:
    if avg_arm_angle < 50:
      counter += 1
      status = 1
      print("Push-up: ", counter)
  else:
      if avg_arm_angle > 160:
        status = 0

  return [counter, status]

"""Pullup function will count the repitions the user is doing depending upon the y coordinate of shoulder is above average y coordinate of elbows """

def PullupUpdated(landmarks, counter, status,  image):
  l_elbow = BodyPartDimensions(landmarks, "LEFT_ELBOW")
  l_shoulder = BodyPartDimensions(landmarks, "LEFT_SHOULDER")
  l_wrist = BodyPartDimensions(landmarks, "LEFT_WRIST")
  left_arm_angle =  calculateAngle(l_shoulder, l_elbow, l_wrist)
  
  
  r_shoulder = BodyPartDimensions(landmarks, "RIGHT_SHOULDER")
  r_elbow = BodyPartDimensions(landmarks, "RIGHT_ELBOW")
  r_wrist = BodyPartDimensions(landmarks, "RIGHT_WRIST")
  right_arm_angle = calculateAngle(r_shoulder, r_elbow, r_wrist)
  avg_arm_angle = (left_arm_angle + right_arm_angle)//2

  # using opencv displaying angle value on image
  cv2.putText(image, "Avg Arm Angle: {}".format(avg_arm_angle), (15,20),
  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA )


    # Render detections
  mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
  mp_drawing.DrawingSpec(color=(240, 128, 128), thickness=3, circle_radius=2), 
  mp_drawing.DrawingSpec(color=(100, 149, 237), thickness=2, circle_radius=2) 
  )               
        
       
  cv2_imshow(image)       
            
  if status == 0:
    if avg_arm_angle < 50:
      counter += 1
      status = 1
      print("Pull-up: ", counter)
  else:
      if avg_arm_angle > 160:
        status = 0

  return [counter, status]

"""Squat function will count the repetition of squat according to the right and left legs angles counted between hip, knee and ankle"""

def squat(landmarks, counter, status,  image):
  r_hip = BodyPartDimensions(landmarks, "RIGHT_HIP")
  r_knee = BodyPartDimensions(landmarks, "RIGHT_KNEE")
  r_ankle = BodyPartDimensions(landmarks, "RIGHT_ANKLE")
  right_leg_angle = calculateAngle(r_hip, r_knee, r_ankle)

  l_hip = BodyPartDimensions(landmarks, "LEFT_HIP")
  l_knee = BodyPartDimensions(landmarks, "LEFT_KNEE")
  l_ankle = BodyPartDimensions(landmarks, "LEFT_ANKLE")
  left_leg_angle = calculateAngle(l_hip, l_knee, l_ankle)

  avg_leg_angle = (left_leg_angle + right_leg_angle) // 2

  # using opencv displaying angle value on image
  cv2.putText(image, "Left Leg: {}".format(left_leg_angle), (15,20),
  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA )
  cv2.putText(image, "Right Leg: {}".format(right_leg_angle), (15,50),
  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA )

    # Render detections
  mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
  mp_drawing.DrawingSpec(color=(240, 128, 128), thickness=3, circle_radius=2), 
  mp_drawing.DrawingSpec(color=(100, 149, 237), thickness=2, circle_radius=2) 
  )

  cv2_imshow(image)
  
  if status == 0:
    if avg_leg_angle < 90:
      counter += 1
      status = 1
      print("squat: ", counter)
  else:
    if avg_leg_angle > 160:
      status = 0

  return [counter, status]
  
  
@app.route('/')
def home(name):
   return render_template('index.html')

@app.route('https://gymtrainer-diploma.herokuapp.com/posted',methods = ['POST'])
def processResult():
  cap = cv2.VideoCapture("/content/sample_data/"+exerciseName+".mp4")
  counter = 0
  status = 0

  ## mediapipe instance and extracting frames and calculating the repetitions of exercise
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    try:
      while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (720, 420))
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
         
        # Extract landmarks
        try:
          landmarks = results.pose_landmarks.landmark
          counter, status = globals()[exerciseName](landmarks, counter, status, image)
        except:
          pass

    except:
    print("end of video")


  print("Total repetition : ", counter)
  cap.release()
  cv2.destroyAllWindows()

    

if __name__ == '__main__':
   app.run(debug = True)
