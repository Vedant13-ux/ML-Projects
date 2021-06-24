import cv2
import mediapipe as mp
import time

from mediapipe.python.solutions.drawing_utils import draw_detection

class FaceDetector():
    def __init__(self):
       self.mpFaceDetection=mp.solutions.face_detection
       self.mpDraw=mp.solutions.drawing_utils
       self.faceDetection=self.mpFaceDetection.FaceDetection(0.75)
    
    def findFace(self,img, draw=True):
        imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results=self.faceDetection.process(imgRGB)
        if results.detections:
            for id, det in enumerate(results.detections):
                h,w,c=img.shape
                bboxC=det.location_data.relative_bounding_box
                bbox=int(bboxC.xmin*w), int(bboxC.ymin*h), int(bboxC.width*w), int(bboxC.height*h)
                cv2.rectangle(img, bbox, (255, 0, 255),2)
                cv2.putText(img, f'{int(det.score[0]*100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    


                       
   

def main():
    pTime=0
    cTime=0
    cap=cv2.VideoCapture(0)
    detector=FaceDetector()
    while True:
        _, frame=cap.read()
        frame=detector.findFace(frame)
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

        cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Frame",frame)
        cv2.waitKey(1)
      


if __name__=="__main__":
    main()