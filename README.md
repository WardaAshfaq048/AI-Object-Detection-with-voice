Here's a clean and professional README for your YOLOv3 object detection Streamlit project, tailored as your AI final lab project:

---

# YOLOv3 Object Detection with Streamlit

This project demonstrates real-time object detection using the YOLOv3 deep learning model integrated with a Streamlit web app. The app captures live video from a webcam, detects objects with Non-Maximum Suppression (NMS) for better accuracy, counts detected objects, and highlights potentially dangerous objects like knives and guns.

## Features

* Real-time video capture and object detection using YOLOv3.
* Applies Non-Maximum Suppression (NMS) to reduce overlapping bounding boxes.
* Counts the number of detected objects by category.
* Highlights the closest detected object based on bounding box area.
* Detects and alerts when dangerous objects (e.g., knife, gun) are present.
* Simple and interactive user interface with Streamlit.

## Installation and Setup

1. **Clone this repository:**

   ```
   git clone <your-repository-url>
   cd <your-project-folder>
   ```

2. **Install dependencies:**

   ```
   pip install -r requirements.txt
   ```

   Required libraries include:

   * streamlit
   * opencv-python
   * numpy

3. **Download YOLOv3 weights and config files:**

   * `yolov3.weights` from [YOLO website](https://pjreddie.com/media/files/yolov3.weights)
   * `yolov3.cfg` from the official YOLO repository
   * `coco.names` file containing class labels

   Place these files in your project directory or update the code paths accordingly.

4. **Run the Streamlit app:**

   ```
   streamlit run your_script_name.py
   ```

5. **Use the app:**

   * Check the "Start Camera" checkbox to activate webcam object detection.
   * View bounding boxes and labels around detected objects.
   * See counts of detected objects.
   * Get alerts if dangerous objects are detected.

## Project Description

This project is part of my AI final lab at COMSATS University Islamabad, Sahiwal campus. It combines computer vision techniques with interactive web deployment to demonstrate practical use of object detection models. The system uses the YOLOv3 model pretrained on the COCO dataset to detect and classify objects in real-time video feeds.


