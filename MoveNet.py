import cv2
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import csv

# Load YOLO model
# net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
# layer_names = net.getLayerNames()
# output_layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
#
#
# # Load the COCO class labels
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]

# Define the CSV file and write the header
csv_file = 'keypoints_data.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    header = ['frame_id', 'person_id']
    for i in range(17):  # Assuming 17 keypoints
        header += [f'keypoint_{i}_x', f'keypoint_{i}_y', f'keypoint_{i}_confidence']
    writer.writerow(header)
frame_id = 0

# Ensure TensorFlow is able to use the GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Loading the model from TensorFlow Hub
model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']


edges = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
    (5, 7): 'm', (7, 9): 'm', (6, 8): 'c', (8, 10): 'c',
    (5, 6): 'y', (5, 11): 'm', (6, 12): 'c', (11, 12): 'y',
    (11, 13): 'm', (13, 15): 'm', (12, 14): 'c', (14, 16): 'c'
}


# Drawing the keypoints on the frame
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 5, (0, 255, 0), -1)


# Drawing connections between keypoints
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)


# Process each detected person in the frame
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)


# def detect_referees(image):
#     height, width, _ = image.shape
#     blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layer_names)
#
#     class_ids = []
#     confidences = []
#     boxes = []
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5 and class_id == classes.index("person"):
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
#
#     indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#
#     # Check if indices is not empty and flatten it if necessary
#     if indices is not None and len(indices) > 0:
#         # Flatten indices if they are in a list of lists
#         indices = [i[0] if isinstance(i, np.ndarray) else i for i in indices]
#         final_boxes = [boxes[i] for i in indices]
#     else:
#         final_boxes = []
#
#     return final_boxes

# Read the video
cap = cv2.VideoCapture('SuperbonvPetrosyan.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)  # Reduce output resolution for better performance
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, size)

confidence_threshold = 0.225



while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (640, 360))  # Adjusted for performance
    image = tf.cast(tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 352, 640), dtype=tf.int32)

    results = movenet(image)
    keypoints = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))

    # Use the updated confidence threshold
    loop_through_people(img, keypoints, edges, confidence_threshold)

    resized_img = cv2.resize(img, size)  # Resize the processed frame back to the output video size
    cv2.imshow('frame', resized_img)
    out.write(resized_img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        for person_id, person_keypoints in enumerate(keypoints):
            row = [frame_id, person_id]
            for kp in np.squeeze(person_keypoints):
                ky, kx, kp_conf = kp
                row += [kx, ky, kp_conf]
            writer.writerow(row)

    frame_id += 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()