# Invisibility Cloak using OpenCV 🧙‍♂️

This project brings the magic of Harry Potter's invisibility cloak to life using Python and computer vision. It utilizes OpenCV and NumPy to perform real-time background subtraction and color detection through a webcam. Initially, the script captures a static background image, and then dynamically detects a specific target color (like a red cloth) frame-by-frame. Wherever the target color is detected, the program seamlessly replaces those pixels with the original background. The result is a magical optical illusion that makes anything behind the cloak appear completely invisible!

## Features
- Real-time computer vision processing using OpenCV
- Color-based foreground replacement (targets red by default)
- Easy to use and visually satisfying illusion

## Prerequisites
Ensure you have Python installed on your system. This project requires:
- `opencv-python`
- `numpy`

## Installation
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/Invisibility-Cloak.git
   cd Invisibility-Cloak
   ```
2. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

## How to Use
1. Have a solid red cloth, blanket, or jacket ready.
2. Run the main script:
   ```bash
   python fast_invisible.py
   ```
3. **Crucial Step**: The moment you run the script, immediately step **out of the frame**. The terminal will count down from 3 to 1 to capture your static room background. 
4. After the terminal says *"[SUCCESS] Background captured!"*, step back into the camera frame with your red cloth. 
5. Dazzle observers as anything wrapped in or hidden behind the cloak becomes completely invisible!

## Exiting the Program
To safely stop the program, make sure you click on the pop-up video window to focus it, and then simply press the **`Esc`** key on your keyboard.
