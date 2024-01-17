import math
from ctypes import cast, POINTER
import cv2
import mediapipe as mp
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)
        thumb = hand.landmark[4]
        index = hand.landmark[8]
        h, w, c = img.shape
        x1, y1 = int(thumb.x * w), int(thumb.y * h)
        x2, y2 = int(index.x * w), int(index.y * h)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        min_length = 30
        max_length = 350
        min_volume = -63.5
        max_volume = 0.0
        volume_percentage = np.interp(length, [min_length, max_length], [0, 100])
        volume_level = np.interp(volume_percentage, [0, 100], [min_volume, max_volume])
        volume.SetMasterVolumeLevel(volume_level, None)

        # Display the volume percentage on the frame
        cv2.putText(img, f'Volume: {int(volume_percentage)} %', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == 32:
        break

cap.release()
cv2.destroyAllWindows()
