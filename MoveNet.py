import cv2
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import csv

# Initialize the CSV file to store keypoints data and write the header
csv_file = 'Keypoint Files/GloryRingTest.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    header = ['frame_id', 'person_id']
    for i in range(17):  # Assuming 17 keypoints per person
        header += [f'keypoint_{i}_x', f'keypoint_{i}_y', f'keypoint_{i}_confidence']
    writer.writerow(header)
frame_id = 0

# Configure TensorFlow to use GPU if available to enhance performance
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load the MoveNet model from TensorFlow Hub
model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

# Define a dictionary for connections between keypoints with their respective color codes
edges = {
    # Connections are defined with a tuple of keypoints and the color as a string
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
    (5, 7): 'm', (7, 9): 'm', (6, 8): 'c', (8, 10): 'c',
    (5, 6): 'y', (5, 11): 'm', (6, 12): 'c', (11, 12): 'y',
    (11, 13): 'm', (13, 15): 'm', (12, 14): 'c', (14, 16): 'c'
}

# Function to draw keypoints on an image
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 5, (0, 255, 0), -1)

# Function to draw connections between keypoints based on the defined edges
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

# Setup video capture and writer
cap = cv2.VideoCapture('GloryRingTest.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('GloryRingTestOutput.mp4', fourcc, 20.0, size)
confidence_threshold = 0.225

# Main loop to process each frame of the video
while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (640, 360))  # Resize for performance optimization
    image = tf.cast(tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 352, 640), dtype=tf.int32)

    # Apply MoveNet model to detect keypoints
    results = movenet(image)
    keypoints = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))

    # Process and draw keypoints and connections on the image
    loop_through_people(img, keypoints, edges, confidence_threshold)

    # Resize processed image to fit video output and display
    resized_img = cv2.resize(img, size)
    cv2.imshow('frame', resized_img)
    out.write(resized_img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    # Append keypoints data to the CSV file
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        for person_id, person_keypoints in enumerate(keypoints):
            row = [frame_id, person_id]
            for kp in np.squeeze(person_keypoints):
                ky, kx, kp_conf = kp
                row += [kx, ky, kp_conf]
            writer.writerow(row)

    frame_id += 1  # Increment frame ID for the next loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Clean up: release video capture and writer, destroy all windows
cap.release()
out.release()
cv2.destroyAllWindows()
