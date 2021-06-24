import cv2
import mediapipe as mp
import time

from mediapipe.python.solutions.drawing_utils import draw_detection
cap=cv2.VideoCapture(0)
pTime=0
cTime=0
mpFaceDetection=mp.solutions.face_detection
mpDraw=mp.solutions.drawing_utils
faceDetection=mpFaceDetection.FaceDetection(0.75)
while True:
    _, frame=cap.read()
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    imgRGB=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results=faceDetection.process(imgRGB)
    if results.detections:
        for id, det in enumerate(results.detections):
            h,w,c=frame.shape
            bboxC=det.location_data.relative_bounding_box
            bbox=int(bboxC.xmin*w), int(bboxC.ymin*h), int(bboxC.width*w), int(bboxC.height*h)
            cv2.rectangle(frame, bbox, (255, 0, 255),2)
            cv2.putText(frame, f'{int(det.score[0]*100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)


    cv2.imshow("Frame",frame)
    cv2.waitKey(1)


    