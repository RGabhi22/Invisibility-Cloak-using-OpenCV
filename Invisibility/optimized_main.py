import cv2
import numpy as np
import os
from imutils.video import FPS

# --- CONFIGURATION ---
weightsPath = "mask-rcnn-coco/frozen_inference_graph.pb"
configPath = "mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

# Skip frames to boost speed (Higher number = Faster but less accurate tracking)
SKIP_FRAMES = 3

expected_confidence = 0.3
threshold = 0.1
kernel = np.ones((5,5), np.uint8)

# Initialize
fps = FPS().start()
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# --- CAMERA SETUP ---
print("[INFO] accessing video stream...")
cap = cv2.VideoCapture(0)

# Try to increase camera clarity (Resolution)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("[INFO] Get out of the frame! Background recording in 3 seconds...")
cv2.waitKey(3000) 

print("[INFO] Capturing background...")
# FIXED: Simple warmup loop (No complex math to avoid errors)
bg = None
for i in range(60):
    ret, bg = cap.read()

if bg is None:
    print("[ERROR] Could not read from camera.")
    exit()

print("[INFO] Background recording done. Step back in!")

# Setup Video Writer
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter('output_final.avi', fourcc, 20, (bg.shape[1], bg.shape[0]), True)

# Variables for optimization
frame_count = 0
last_detected_mask = None 

# --- MAIN LOOP ---
while True:
    grabbed, frame = cap.read()
    if not grabbed:
        break
    
    (H, W) = frame.shape[:2]
    
    # Only run the Heavy AI every 'SKIP_FRAMES' times
    if frame_count % SKIP_FRAMES == 0:
        blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
        net.setInput(blob)
        (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])

        full_frame_mask = np.zeros((H, W), dtype="uint8")

        for i in range(0, boxes.shape[2]):
            classID = int(boxes[0, 0, i, 1])
            confidence = boxes[0, 0, i, 2]

            # Class 0 is usually person in this model
            if classID == 0 and confidence > expected_confidence:
                box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                
                boxW = endX - startX
                boxH = endY - startY
                
                if boxW > 0 and boxH > 0:
                    mask = masks[i, classID]
                    mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_CUBIC)
                    mask = (mask > threshold)
                    
                    bwmask = np.array(mask, dtype=np.uint8) * 255
                    bwmask = cv2.dilate(bwmask, kernel, iterations=1)

                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(W, endX)
                    endY = min(H, endY)

                    mask_h, mask_w = bwmask.shape
                    if (endY - startY) == mask_h and (endX - startX) == mask_w:
                        full_frame_mask[startY:endY, startX:endX] = bwmask

        last_detected_mask = full_frame_mask

    # Use the memory (last_detected_mask) to apply the effect
    if last_detected_mask is not None:
        # Wherever the mask is white (255), replace frame pixels with background pixels
        frame[np.where(last_detected_mask == 255)] = bg[np.where(last_detected_mask == 255)]

    cv2.imshow("Invisible Man (Optimized)", frame)
    
    if cv2.waitKey(1) == 27: # ESC to exit
        break

    writer.write(frame)
    fps.update()
    frame_count += 1

fps.stop()
cap.release()
writer.release()
cv2.destroyAllWindows()