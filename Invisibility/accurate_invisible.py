import cv2
import numpy as np
import time

# --- CONFIGURATION ---
cap = cv2.VideoCapture(0)

# Set to HD resolution for better detection
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
# Capture more frames (60) for a smoother background
for i in range(60):
    ret, background = cap.read()

if background is None:
    print("[ERROR] Camera not working")
    exit()

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

    # 3. Define WIDER RED color range
    # I have lowered the Saturation (S) and Value (V) to catch shadows on your shirt
    
    # Range 1 (0-10) -> Extended to 0-20 to catch orange-reds
    lower_red1 = np.array([0, 70, 50])     # Lowered S and V to catch dark/dull red
    upper_red1 = np.array([20, 255, 255])
    
    # Range 2 (170-180) -> Extended to 160-180 to catch pink-reds
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # 4. Create Mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2 

    # 5. Clean Mask (MAJOR IMPROVEMENT HERE)
    # Using a larger kernel (5x5) to fill in the "holes" in your shirt
    kernel = np.ones((5, 5), np.uint8)
    
    # 'OPEN' removes white noise dots from the background
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    # 'DILATE' expands the red area slightly to cover the edges
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # 6. Smooth the mask edges (Anti-aliasing)
    # This blurs the mask slightly so edges don't look jagged
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # 7. Invert Mask (Everything NOT red)
    mask_inv = cv2.bitwise_not(mask)

    # 8. Apply Magic
    # Background part (Where the cloak is)
    res1 = cv2.bitwise_and(background, background, mask=mask)
    # Frame part (Where the cloak is NOT)
    res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)
    
    # Combine
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # 9. Show Output
    cv2.imshow('Accurate Invisible Cloak', final_output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()