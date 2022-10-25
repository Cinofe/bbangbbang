import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

Right_Eyes = [362, 382, 381, 380, 374, 373, 390,
              249, 263, 466, 388, 387, 386, 385, 384, 398]
Left_Eyes = [33, 7, 163, 144, 145, 153, 154,
             155, 133, 173, 157, 158, 159, 160, 161, 246]


def return_Eye(facePos, eyePos):
    minx, miny, maxx, maxy = w, h, 0, 0
    for e in eyePos:
        lx = int(facePos[e].x * w)
        ly = int(facePos[e].y * h)
        if lx < minx:
            minx = lx
        if ly < miny:
            miny = ly
        if lx > maxx:
            maxx = lx
        if ly > maxy:
            maxy = ly
    return image[miny-30:maxy+30, minx-30:maxx+30].copy()


drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        max_num_faces=1,
        min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        h, w = image.shape[:2]
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_pos = face_landmarks.landmark

                rightEye = return_Eye(face_pos, Right_Eyes)
                leftEye = return_Eye(face_pos, Left_Eyes)

        cv2.imshow('MediaPipe FaceMesh', image)
        cv2.imshow("right Eye", rightEye)
        cv2.imshow("left Eye", leftEye)

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()

#159, 145
#386, 374
