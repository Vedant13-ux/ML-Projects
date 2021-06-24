import cv2
import mediapipe as mp
import time

class PoseDetector():
    def __init__(self):
        self.mpPose=mp.solutions.pose
        self.mpDraw=mp.solutions.drawing_utils
        self.pose=self.mpPose.Pose()
    
    def findHands(self,img, draw=True):
        imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results=self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img
    
    def findPosition(self, img, draw=True):
        lmlsit=[]
        if self.results.pose_landmarks:
            for id, lm, in enumerate(self.results.pose_landmarks.landmark):
                h, w ,c=img.shape
                cx, cy=int(lm.x*w), int(lm.y*h)
                lmlsit.append([id, cx, cy])
                if draw:
                    cv2.circle(img,(cx,cy), 15, (255,0,255), cv2.FILLED)
        
        return lmlsit

                       
   

def main():
    pTime=0
    cTime=0
    cap=cv2.VideoCapture(0)
    detector=PoseDetector()
    while True:
        _, frame=cap.read()
        frame=detector.findHands(frame)
        lmlist=detector.findPosition(frame)
        if len(lmlist) !=0:
            print(lmlist[4])    
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

        cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Frame",frame)
        cv2.waitKey(1)
      


if __name__=="__main__":
    main()