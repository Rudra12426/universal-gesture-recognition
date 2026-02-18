# if you facing any error in this use jupyter notebok otherwise it will run. 
"""
Universal Gesture Recognition System
Author: Rudra Pratap
Description: Real-time hand gesture recognition using OpenCV.
Detects finger count and provides voice feedback.
"""

import cv2
import numpy as np
import pyttsx3


def initialize_speech_engine():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    return engine


def main():
    engine = initialize_speech_engine()

    cap = cv2.VideoCapture(0)

    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])

    emojis = {0:"✊", 1:"☝️", 2:"✌️", 3:"🤟", 4:"🖐️", 5:"🖖"}

    prev_finger_count = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)
        mask_red  = cv2.inRange(hsv, lower_red, upper_red)
        mask = cv2.bitwise_or(mask_skin, mask_red)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        finger_count = 0

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(largest_contour) > 1000:
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 3)

                hull = cv2.convexHull(largest_contour, returnPoints=False)

                if hull is not None and len(hull) > 3:
                    defects = cv2.convexityDefects(largest_contour, hull)

                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(largest_contour[s][0])
                            end = tuple(largest_contour[e][0])
                            far = tuple(largest_contour[f][0])

                            a = np.linalg.norm(np.array(end) - np.array(start))
                            b = np.linalg.norm(np.array(far) - np.array(start))
                            c = np.linalg.norm(np.array(end) - np.array(far))

                            angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))

                            if angle <= np.pi / 2:
                                finger_count += 1

        cv2.putText(frame, f"Fingers: {finger_count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.putText(frame, emojis.get(finger_count, ""), (200, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 4)

        if finger_count != prev_finger_count:
            engine.say(str(finger_count))
            engine.runAndWait()
            prev_finger_count = finger_count

        cv2.imshow("Universal Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



    main()
