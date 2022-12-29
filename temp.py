from cvzone.HandTrackingModule import HandDetector
import cv2  

cap_cam = cv2.VideoCapture(0)

cap_cam.set(3, 800)
cap_cam.set(4, 600)

detector = HandDetector(maxHands = 2, detectionCon = 1)

while True:
    sucess, cam_img = cap_cam.read()
    cam_img = detector.findHands(cam_img)
    lm_list, bboxInfo = detector.findPosition(cam_img)

    if lm_list:
        l,_,_ = detector.findDistance(8, 12, cam_img, draw = False)
        #print(l)
        print(l)

    cv2.imshow("image", cam_img)
    if cv2.waitKey(1) == ord('q'): 
        break