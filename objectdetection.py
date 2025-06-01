import streamlit as st
import cv2
import numpy as np

# Load YOLO model and classes
@st.cache_resource
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open(r"D:\Downloads\coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getUnconnectedOutLayersNames()
    return net, classes, layer_names

net, classes, layer_names = load_yolo()

# Define dangerous objects (customizable)
dangerous_objects = ["knife", "gun"]

st.title("YOLOv3 Object Detection with NMS and Object Counting")
run = st.checkbox('Start Camera')

FRAME_WINDOW = st.image([])
closest_obj_text = st.empty()
danger_text = st.empty()
count_text = st.empty()

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame")
        break

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    max_area = 0
    closest_object_label = ""
    danger_detected = False
    object_counts = {}

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            confidence = confidences[i]

            label = f"{classes[class_id]}: {int(confidence * 100)}%"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            class_name = classes[class_id]
            object_counts[class_name] = object_counts.get(class_name, 0) + 1

            area = w * h
            if area > max_area:
                max_area = area
                closest_object_label = class_name

            if class_name.lower() in dangerous_objects:
                danger_detected = True

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame_rgb)

    if closest_object_label:
        closest_obj_text.markdown(f"### Closest Object: `{closest_object_label}`")

    if danger_detected:
        danger_text.markdown("## ⚠️ **Dangerous Object Detected!**")
    else:
        danger_text.markdown("")

    if object_counts:
        count_text.markdown("### Objects detected:")
        for obj, count in object_counts.items():
            count_text.markdown(f"- {obj}: {count}")
    else:
        count_text.markdown("No objects detected.")

cap.release()
