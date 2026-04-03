import os
import urllib.request
import tarfile
import shutil

# --- CONFIGURATION ---
TARGET_FOLDER = "mask-rcnn-coco"
WEIGHTS_URL = "http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz"
CONFIG_URL = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

# File names
TAR_FILENAME = "mask_rcnn_model.tar.gz"
WEIGHTS_FILE_INSIDE_TAR = "mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"
FINAL_WEIGHTS_NAME = "frozen_inference_graph.pb"
FINAL_CONFIG_NAME = "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

def download_file(url, filename):
    print(f"[INFO] Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print("[SUCCESS] Download complete.")
    except Exception as e:
        print(f"[ERROR] Failed to download {url}. Error: {e}")
        exit()

def main():
    # 1. Create the folder if it doesn't exist
    if not os.path.exists(TARGET_FOLDER):
        os.makedirs(TARGET_FOLDER)
        print(f"[INFO] Created folder: {TARGET_FOLDER}")

    # 2. Download and Extract Weights
    print("\n--- STEP 1: PROCESSING WEIGHTS ---")
    download_file(WEIGHTS_URL, TAR_FILENAME)
    
    print("[INFO] Extracting model from archive...")
    try:
        with tarfile.open(TAR_FILENAME, "r:gz") as tar:
            # We only want the specific .pb file
            member = tar.getmember(WEIGHTS_FILE_INSIDE_TAR)
            member.name = FINAL_WEIGHTS_NAME # Rename on extraction
            tar.extract(member, path=TARGET_FOLDER)
        print("[SUCCESS] Extracted 'frozen_inference_graph.pb' to folder.")
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        exit()

    # 3. Download Config
    print("\n--- STEP 2: PROCESSING CONFIG ---")
    config_path = os.path.join(TARGET_FOLDER, FINAL_CONFIG_NAME)
    download_file(CONFIG_URL, config_path)
    print(f"[SUCCESS] Config saved to {config_path}")

    # 4. Cleanup
    print("\n--- CLEANUP ---")
    if os.path.exists(TAR_FILENAME):
        os.remove(TAR_FILENAME)
        print("[INFO] Removed temporary files.")

    print("\n" + "="*40)
    print(" ALL DONE! You can now run main.py")
    print("="*40)

if __name__ == "__main__":
    main()