
import numpy as np
import cv2
import dlib
import imutils
from imutils import face_utils

from math import hypot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D


cap = cv2.VideoCapture (0)
face_cascade = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")

# ----表情認識---------------------------------------------- #  
#  モデルを作成する
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# 感情をmodelより取得
model.load_weights('model/model.h5')

# openCLの使用と不要なロギングメッセージを防ぐ
cv2.ocl.setUseOpenCL(False)

# 感情を割り当て（アルファベット順）
emotion = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# カスケードファイル呼び出し
face_cascade_default = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
# ----END 表情認識---------------------------------------------- #


# VideoCapture オブジェクトを取得します
#DEVICE_ID = 0 #ID 0は標準web cam
capture = cv2.VideoCapture(0)#dlibの学習済みデータの読み込み

detector = dlib.get_frontal_face_detector() #顔検出器の呼び出し。ただ顔だけを検出する。
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat") #顔から目鼻などランドマークを出力する

def midpoint(p1 ,p2): #中点
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN
ret, frame = capture.read()
def get_blinking_ratio(eye_points, facial_landmarks): #点滅＿比率
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # 目枠（赤）
    #cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)#
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio

data = []       
bli=0
aaa=0

def Ai_count():
    print("")

def Ai(a,b,c,d,e,f,g,h,i):
    global root
    global canvas1
    global bli

    prediction=0
    face_r=0
    face_c=0
    face_l=0
    eye_r=0
    eye_c=0
    eye_l=0
    eye_u=0
    nod=0
    smail=0

    while(True): #カメラから連続で画像を取得する

        ret, frame = capture.read() #カメラからキャプチャしてframeに１コマ分の画像データを入れる

        frame = imutils.resize(frame) #frameの画像の表示サイズを整える
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #gray scaleに変換する
        rects = detector(gray, 0) #grayから顔を検出
        image_points = None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        recognized_faces = face_cascade_default.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=7)

        # 顔認証
        for (x, y, w, hh) in recognized_faces:
            roi_gray = gray[y:y + hh, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            
        facial_expression = int(np.argmax(prediction))

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            for (x, y) in shape: #顔全体の68箇所のランドマークをプロット
                cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

            image_points = np.array([
                    tuple(shape[30]),   # 31 鼻頭
                    tuple(shape[21]),   # 22 眉左
                    tuple(shape[22]),   # 23 眉右
                    tuple(shape[39]),   # 40 目頭左
                    tuple(shape[42]),   # 43 目頭右
                    tuple(shape[31]),   # 32 鼻左
                    tuple(shape[35]),   # 36 鼻右
                    tuple(shape[48]),   # 49 口左
                    tuple(shape[54]),   # 55 口右
                    tuple(shape[57]),   # 58 口下
                    tuple(shape[8]),    # 9  顎 
                    ],dtype='double')

        if len(rects) > 0:
            model_points = np.array([
                    (0.0,0.0,0.0),        # [30] 31 鼻頭
                    (-30.0,-125.0,-30.0), # [21] 22 眉左
                    (30.0,-125.0,-30.0),  # [22] 23 眉右
                    (-60.0,-70.0,-60.0),  # [39] 40 目頭左
                    (60.0,-70.0,-60.0),   # [42] 43 目頭右
                    (-40.0,40.0,-50.0),   # [31] 32 鼻左
                    (40.0,40.0,-50.0),    # [35] 36 鼻右
                    (-70.0,130.0,-100.0), # [48] 49 口左
                    (70.0,130.0,-100.0),  # [54] 55 口右
                    (0.0,158.0,-10.0),    # [57] 58 口下
                    (0.0,250.0,-50.0)     # [8 ] 9  顎 
                    ])

            
            landmarks = predictor(gray, rect)
            # Detect blinking
            left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
            right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
            

            
            gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
            gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
            gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

            size = frame.shape

            focal_length = size[1]
            center = (size[1] // 2, size[0] // 2) #顔の中心座標

            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype='double')

            dist_coeffs = np.zeros((4, 1))

            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                        dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            #回転行列とヤコビアン
            (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
            mat = np.hstack((rotation_matrix, translation_vector))

            #yaw,pitch,rollの取り出し
            (_, _, _, _, _, _, eulerAngles) = cv2.decomposeProjectionMatrix(mat)
            yaw = eulerAngles[1]
            pitch = eulerAngles[0]
            roll = eulerAngles[2]

            if gaze_ratio <= 0.85:
                eye_r=1
                eye_c=0
                eye_l=0

            elif 0.85 < gaze_ratio < 1.0:
                eye_r=0
                eye_c=1
                eye_l=0
            elif gaze_ratio >= 1.0:
                eye_r=0
                eye_c=0
                eye_l=1
            
            if 4.0 < blinking_ratio < 6.0:
                bli+=1
                if bli>2:
                    cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))
                    eye_u=1
                    eye_r=0
                    eye_c=0
                    eye_l=0
            else:
                bli=0

            if pitch<-10:
                if facial_expression != 3:
                    nod=1
                elif facial_expression == 3:
                    nod=0

            if yaw<-10:
                face_r=1
                face_c=0
                face_l=0
            elif yaw>10:
                face_r=0
                face_c=0
                face_l=1
            else:
                face_r=0
                face_c=1
                face_l=0

            #笑顔
            if facial_expression == 3:
                smail=1
            else:
                smail=0

            (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector,translation_vector, camera_matrix, dist_coeffs)
            
            #計算に使用した点のプロット/顔方向のベクトルの表示
            for p in image_points:
                cv2.circle(frame, (int(p[0]), int(p[1])), 5, (0, 215, 255), -1)

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            cv2.arrowedLine(frame, p1, p2, (255, 0, 0), 2)

        a+=face_r
        b+=face_l
        c+=face_c
        d+=eye_r
        e+=eye_l
        f+=eye_c
        g+=eye_u
        h+=nod
        i+=smail
        
        return a,b,c,d,e,f,g,h,i


