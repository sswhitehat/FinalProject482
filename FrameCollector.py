import cv2
import os


def extract_frames(video_path, output_dir):
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the video
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while cap.isOpened():
        # Read the next frame from the video
        ret, frame = cap.read()

        # Check if frame was read successfully
        if not ret:
            break  # Exit the loop if there are no frames left to read

        # Construct the output file path
        frame_file = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")

        # Save the current frame as an image file
        cv2.imwrite(frame_file, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Extracted {frame_count} frames and saved to directory: {output_dir}")


# Example usage
video_path = 'SuperbonvPetrosyan.mp4'
output_dir = 'H:/SuperbonvPetrosyan'
extract_frames(video_path, output_dir)