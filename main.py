import cv2
from cv2 import FONT_HERSHEY_COMPLEX, FONT_HERSHEY_PLAIN
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import numpy as np
import time
from Arm_Lib import Arm_Device

Arm = Arm_Device()
time.sleep(.1)

time_1 = 500
time_2 = 1000
time_sleep = 0.5

idx = 0

gesture = {
    0:'fist', 1:'one', 3:'three', 4:'four', 5:'five', 7:'rock', 8:'spiderman', 9:'V^^V', 10:'ok',
}

rps_gesture = {0:'rock', 5:'paper', 9:'V^^V'}


cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=2, detectionCon=1)

#Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#mediapipe 중 손의 관절 위치를 인식할 수 있는 모델 초기화
hands = mp_hands.Hands(
    max_num_hands = 2, # 몇 개의 손을 인식할 것이냐
    min_detection_confidence = 0.5, # 탐지 임계치
    min_tracking_confidence = 0.5 # 추정 임계치
)
 
file = np.genfromtxt('gesture_train.csv', delimiter=',') # 파일을 읽어온다
angle = file[:, :-1].astype(np.float32) # 0번 idx부터 마지막 idx전까지 사용하기
label = file[:, -1].astype(np.float32) # 마지막 idx만 사용하기

knn = cv2.ml.KNearest_create() # knn 모델 초기화
knn.train(angle, cv2.ml.ROW_SAMPLE, label) # knn 학습

def dof(a, b, c, d, e, f):
    Arm.Arm_serial_servo_write(1, a, 500)
    time.sleep(2)
    Arm.Arm_serial_servo_write(2, b, 1000)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(3, c, 1000)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(4, d, 1000)
    time.sleep(1.5)
    Arm.Arm_serial_servo_write(5, e, 1000)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(6, f, 1000)
    time.sleep(2)

def dance():
    Arm.Arm_serial_servo_write(2, 180-120, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(3, 120, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(4, 60, time_1)
    time.sleep(time_sleep)

    Arm.Arm_serial_servo_write(2, 180-135, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(3, 135, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(4, 45, time_1)
    time.sleep(time_sleep)

    Arm.Arm_serial_servo_write(2, 180-120, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(3, 120, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(4, 60, time_1)
    time.sleep(time_sleep)

    Arm.Arm_serial_servo_write(2, 90, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(3, 90, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(4, 90, time_1)
    time.sleep(time_sleep)

    Arm.Arm_serial_servo_write(2, 180-80, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(3, 80, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(4, 80, time_1)
    time.sleep(time_sleep)


    Arm.Arm_serial_servo_write(2, 180-60, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(3, 60, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(4, 60, time_1)
    time.sleep(time_sleep)

    Arm.Arm_serial_servo_write(2, 180-45, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(3, 45, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(4, 45, time_1)
    time.sleep(time_sleep)

    Arm.Arm_serial_servo_write(2, 90, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(3, 90, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(4, 90, time_1)
    time.sleep(.001)
    time.sleep(time_sleep)

    Arm.Arm_serial_servo_write(4, 20, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(6, 150, time_1)
    time.sleep(.001)
    time.sleep(time_sleep)

    Arm.Arm_serial_servo_write(4, 90, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(6, 90, time_1)
    time.sleep(time_sleep)

    Arm.Arm_serial_servo_write(4, 20, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(6, 150, time_1)
    time.sleep(time_sleep)

    Arm.Arm_serial_servo_write(4, 90, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(6, 90, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(1, 0, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(5, 0, time_1)
    time.sleep(time_sleep)

    Arm.Arm_serial_servo_write(3, 180, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(4, 0, time_1)
    time.sleep(time_sleep)

    Arm.Arm_serial_servo_write(6, 180, time_1)
    time.sleep(time_sleep)

    Arm.Arm_serial_servo_write(6, 0, time_2)
    time.sleep(time_sleep)

    Arm.Arm_serial_servo_write(6, 90, time_2)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(1, 90, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(5, 90, time_1)
    time.sleep(time_sleep)

    Arm.Arm_serial_servo_write(3, 90, time_1)
    time.sleep(.001)
    Arm.Arm_serial_servo_write(4, 90, time_1)
    time.sleep(time_sleep)

def main():
    start_dof =0
    if(start_dof == 0):
            Arm.Arm_serial_servo_write6(90, 90, 30, 65, 90, 90, 1000)
            time.sleep(.001)
            Arm.Arm_serial_servo_write(1, 45, 1000)
            time.sleep(.001)
            Arm.Arm_serial_servo_write(3, 30, 1000)
            time.sleep(.001)
            Arm.Arm_serial_servo_write(4, 65, 1000)
            time.sleep(1.5)
            Arm.Arm_serial_servo_write(5, 90, 1000)
            time.sleep(.001)
            Arm.Arm_serial_servo_write(1, 135, 1000)
            time.sleep(.001)
            Arm.Arm_serial_servo_write(3, 30, 1000)
            time.sleep(.001)
            Arm.Arm_serial_servo_write(4, 65, 1000)
            time.sleep(1.5)
            Arm.Arm_serial_servo_write(5, 90, 1000)
            time.sleep(.001)
            Arm.Arm_serial_servo_write(1, 45, 1000)
            time.sleep(.001)
            Arm.Arm_serial_servo_write(3, 30, 1000)
            time.sleep(.001)
            Arm.Arm_serial_servo_write(4, 65, 1000)
            time.sleep(1.5)
            Arm.Arm_serial_servo_write(5, 90, 1000)
            time.sleep(.001)
            Arm.Arm_serial_servo_write6(90, 90, 30, 65, 90, 90, 1000)
            time.sleep(.001)
            start_dof = 1

    while True:
        order = input()
        if(order == "k"):        
            Arm.Arm_serial_servo_write6(90, 90, 30, 65, 90, 90, 1000)
            time.sleep(.001)

            #0은 도프 기준 오른쪽 180은 왼쪽
            a, b, c, d, e, f = map(float, input().split())

            if(a == "pass"):
                a = 90
            if(b == "pass"):
                b = 90
            if(c == "pass"):
                c = 30
            if(d == "pass"):
                d = 65
            if(e == "pass"):
                e = 90
            if(f == "pass"):
                f = 90
    
            #0, 86, 0, 0, 90, 155
            Arm.Arm_serial_servo_write(1, a, 500)
            time.sleep(2)
            Arm.Arm_serial_servo_write(2, b, 1000)
            time.sleep(.001)
            Arm.Arm_serial_servo_write(3, c, 1000)
            time.sleep(.001)
            Arm.Arm_serial_servo_write(4, d, 1000)
            time.sleep(1.5)
            Arm.Arm_serial_servo_write(5, e, 1000)
            time.sleep(.001)
            Arm.Arm_serial_servo_write(6, f, 1000)
            time.sleep(2)
            Arm.Arm_serial_servo_write6(90, 90, 30, 65, 90, 155, 1000)
            time.sleep(5)
            Arm.Arm_serial_servo_write(6, 90, 100)
            time.sleep(1)
        elif(order == "c"):
            cout = 0
            while cap.isOpened():
                ret, img = cap.read()
                if not ret: break

                img = cv2.flip(img, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                result = hands.process(img) # 프레임에서 손, 관절의 위치를 탐색한다.

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR )
                
                if result.multi_hand_landmarks is not None: # 만약 손이 정상적으로 인식 되었을 때
                    for res in result.multi_hand_landmarks: # 여러 손일 경우 루프를 사용한다.
                        joint = np.zeros((21,3))

                        for j, lm in enumerate(res.landmark): # 21개의 랜드마크를 한 점씩 반복문을 사용해서 처리한다.
                            joint[j] = [lm.x, lm.y, lm.z]
                        
                        # 관절 사이의 각도를 계산한다. 0, 86, 0, 0, 90, 155
                        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint

                        v = v2 - v1 # 팔목과 각 손가락 관절 사이의 벡터를 구한다.

                        v = v / np.expand_dims(np.linalg.norm(v, axis=1), axis=-1) # 단위벡터를 구한다.(벡터/벡터의 길이) 
                        
                        angle = np.arccos(np.einsum('nt,nt->n',
                            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # 단위 벡터를 내적한 값에 accos 값을 구하면 관절 사이의 각도를 구할 수 있다.

                        angle = np.degrees(angle) # 라디안 -> 도 
                        angle = np.expand_dims(angle.astype(np.float32), axis=0) # 머신러닝 모델에 넣어서 추론할 때는 항상 맨 앞 차원 하나를 추가한다.
                        
                        # 제스처 추론
                        _, results, _, _ = knn.findNearest(angle, 3)
                        
                        idx = int(results[0][0])
                        
                        if idx in gesture.keys():
                            gesture_name = gesture[idx]
                            cv2.putText(img, text=gesture_name, org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255),thickness=2)

                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS) # 손의 관절을 프레임에 그린다.
                        
                        temp = 100
                        if(temp != idx):
                            temp = idx
                            cout += 1
                            if(cout == 11):
                                cout = 0

                        if(temp == 5 and cout == 10):
                            dof(90, 180, 0, 0, 90, 180)
                            dof(90, 30, 90, 90, 90, 180)
                            Arm.Arm_serial_servo_write6(90, 90, 30, 65, 90, 90, 1000)
                            time.sleep(.001)

                        elif(temp == 7 and cout == 10):
                            dance()
                            Arm.Arm_serial_servo_write6(90, 90, 30, 65, 90, 90, 1000)
                            time.sleep(.001)

                        elif(temp == 10 and cout == 10):
                            dof(0, 86, 0, 0, 90, 155)
                            Arm.Arm_serial_servo_write6(90, 90, 30, 65, 90, 155, 1000)
                            time.sleep(5)
                            Arm.Arm_serial_servo_write(6, 90, 100)
                            time.sleep(1)
                            
                        elif(temp==9 and cout == 10):
                            detector = HandDetector(maxHands = 2, detectionCon = 1)

                            while True:
                                sucess, cam_img = cap.read()
                                cam_img = detector.findHands(cam_img)
                                lm_list, bboxInfo = detector.findPosition(cam_img)

                                if lm_list:
                                    l,_,_ = detector.findDistance(8, 12, cam_img, draw = False)
                                    #print(l)
                                    print(lm_list[8][0]-lm_list[8][1])

                                    finger = lm_list[8][0]-lm_list[8][1]
                                    
                                    if(l < 50 and l > 35):
                                        Arm.Arm_serial_servo_write(1, 100, 1000)
                                        time.sleep(.001)
                                    else:
                                        Arm.Arm_serial_servo_write(1, 80, 1000)
                                        time.sleep(.001)

                                    if(finger < 300):
                                        # 모으기
                                        Arm.Arm_serial_servo_write(6, 155, 1000)
                                        time.sleep(.001)
                                    else:
                                        # 벌리기
                                        Arm.Arm_serial_servo_write(6, 90, 1000)
                                        time.sleep(.001)

                                cv2.imshow('image', cam_img)
                                if cv2.waitKey(1) == ord('q'): 
                                    cv2.destroyWindow("image")
                                    break

                cv2.imshow('result', img)

                if cv2.waitKey(1) == ord('q'): 
                    cv2.destroyWindow("result")
                    break

 
 
try :
    main()
except KeyboardInterrupt:
    print(" Program closed! ")
    pass