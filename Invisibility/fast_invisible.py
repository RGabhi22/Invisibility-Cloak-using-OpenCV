import cv2
import numpy as np
import time

# --- CONFIGURATION ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("[INFO] Camera starting... Get your Red Cloth ready!")
time.sleep(2)

# --- BACKGROUND CAPTURE ---
print("--------------------------------------------------")
print("[ACTION REQUIRED] Move out of the frame! (Capturing background)")
print("3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1...")
time.sleep(1)
print("--------------------------------------------------")

background = 0
for i in range(30):
    ret, background = cap.read()

# Flip the background
background = cv2.flip(background, 1)

print("[SUCCESS] Background captured! You can enter the frame now.")

# --- MAIN LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Flip the current frame
    frame = cv2.flip(frame, 1)

    # 2. Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 3. Define RED color range
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # 4. Create Mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2 

    # 5. Clean Mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # 6. Invert Mask
    mask_inv = cv2.bitwise_not(mask)

    # 7. Apply Magic
    res1 = cv2.bitwise_and(background, background, mask=mask)
    res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)
    
    # FIXED VARIABLE NAME HERE (final_output)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # 8. Show Output
    cv2.imshow(' Invisible Cloak', final_output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()