import argparse
import sys
import os

try:
    import cv2
except ModuleNotFoundError:
    print("[ERROR] Missing dependency: cv2. Install it with: pip install opencv-python")
    sys.exit(1)

import numpy as np
from imutils.video import FPS


def parse_args():
    parser = argparse.ArgumentParser(description="Invisible Man using Mask R-CNN background replacement")
    parser.add_argument("--source", default="0", help="Camera ID or video file path (default: 0)")
    parser.add_argument("--output", default="output.avi", help="Output video file")
    parser.add_argument("--weights", default="mask-rcnn-coco/frozen_inference_graph.pb", help="Mask R-CNN weights path")
    parser.add_argument("--config", default="mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt", help="Mask R-CNN config path")
    parser.add_argument("--confidence", type=float, default=0.3, help="Minimum detection confidence")
    parser.add_argument("--threshold", type=float, default=0.1, help="Mask threshold")
    parser.add_argument("--no-display", action="store_true", help="Do not show display window")
    parser.add_argument("--log", default=None, help="Optional log file path")
    return parser.parse_args()


def open_video_source(source):
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def main():
    args = parse_args()

    weightsPath = args.weights
    configPath = args.config
    expected_confidence = args.confidence
    threshold = args.threshold
    kernel = np.ones((5, 5), np.uint8)

    for path in (weightsPath, configPath):
        if not os.path.exists(path):
            print(f"[ERROR] Missing file: {path}")
            sys.exit(1)

    print("[INFO] Loading Mask R-CNN model...")
    net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

    print("[INFO] accessing video stream...")
    cap = open_video_source(args.source)
    if not cap.isOpened():
        print("[ERROR] Could not open video source. Check source argument.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] Get out of the frame! Background recording in 3 seconds...")
    cv2.waitKey(3000)

    print("[INFO] Capturing background...")
    bg = None
    for _ in range(60):
        ret, bg = cap.read()
        if not ret:
            continue

    if bg is None:
        print("[ERROR] Camera could not read background.")
        cap.release()
        sys.exit(1)

    (H, W) = bg.shape[:2]
    print("[INFO] Background recording done. Step back in!")

    fps = FPS().start()
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(args.output, fourcc, 20, (W, H), True)

    frame_count = 0
    person_replaced = 0

    log_file = None
    if args.log:
        log_file = open(args.log, "a", encoding="utf-8")

    try:
        while True:
            grabbed, frame = cap.read()
            if not grabbed:
                break

            frame_count += 1

            blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
            net.setInput(blob)

            try:
                (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
            except cv2.error as e:
                print(f"[ERROR] Model inference failed: {e}")
                break

            for i in range(0, boxes.shape[2]):
                classID = int(boxes[0, 0, i, 1])
                confidence = float(boxes[0, 0, i, 2])

                if classID != 0 or confidence <= expected_confidence:
                    continue

                box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                startX = max(0, min(W - 1, startX))
                startY = max(0, min(H - 1, startY))
                endX = max(0, min(W, endX))
                endY = max(0, min(H, endY))

                boxW = endX - startX
                boxH = endY - startY

                if boxW <= 0 or boxH <= 0:
                    continue

                mask = masks[i, classID]
                mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_CUBIC)
                mask = (mask > threshold).astype("uint8") * 255
                mask = cv2.dilate(mask, kernel, iterations=1)

                roi = frame[startY:endY, startX:endX]
                bg_roi = bg[startY:endY, startX:endX]

                if roi.shape[:2] != mask.shape:
                    continue

                person_replaced += 1
                inv_mask = cv2.bitwise_not(mask)
                for c in range(3):
                    roi[:, :, c] = cv2.bitwise_and(roi[:, :, c], inv_mask)
                    roi[:, :, c] = cv2.bitwise_or(roi[:, :, c], cv2.bitwise_and(bg_roi[:, :, c], mask))

                frame[startY:endY, startX:endX] = roi

            if not args.no_display:
                cv2.imshow("Invisible Man (Main)", frame)

            if cv2.waitKey(1) == 27:
                break

            writer.write(frame)
            fps.update()

            if frame_count % 100 == 0:
                fps_text = fps.fps()
                status_msg = f"[INFO] Frames: {frame_count}, Replacements: {person_replaced}, FPS: {fps_text:.2f}"
                print(status_msg)
                if log_file:
                    log_file.write(status_msg + "\n")

    except KeyboardInterrupt:
        print("[INFO] Terminated by user.")
    except Exception as e:
        print(f"[ERROR] Unexpected failure: {e}")

    finally:
        fps.stop()
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        if log_file:
            log_file.write(f"Total frames: {frame_count}, replacements: {person_replaced}\n")
            log_file.close()

        print(f"[INFO] Total frames processed: {frame_count}")
        print(f"[INFO] Person-region replacements: {person_replaced}")


if __name__ == "__main__":
    main()