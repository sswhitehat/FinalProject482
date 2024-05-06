import cv2
import pandas as pd

def overlay_strikes(video_path, csv_path, output_path):
    # Load strike data from a CSV file into a DataFrame
    strikes = pd.read_csv(csv_path)
    # Convert 'Frame Number' column from tensor format to integer by stripping extraneous text
    strikes['Frame Number'] = strikes['Frame Number'].apply(lambda x: int(str(x).replace('tensor(', '').replace(')', '')))

    # Mapping of specific strikes to general categories
    strike_categories = {
        'Jab': 'Punch', 'Cross': 'Punch', 'Hook': 'Punch', 'Upper': 'Punch',
        'Leg Kick': 'Kick', 'Body Kick': 'Kick', 'High Kick': 'Kick'
    }

    # Initialize counts for each strike category
    strike_counts = {'Punch': 0, 'Kick': 0}

    # Initialize video capture and read properties
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec, here 'mp4v' is used for compatibility with .mp4 files
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup the output video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0  # Initialize frame counter
    last_frame = -1  # Variable to keep track of the last processed frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no more frames are available

        # Reset strike counts if we are processing a new frame
        if last_frame != frame_count:
            seen_strikes_this_frame = set()

        # Filter strikes data for the current frame
        current_strikes = strikes[strikes['Frame Number'] == frame_count]
        for _, strike_row in current_strikes.iterrows():
            strike_name = strike_row['Predicted Strike']
            # Update strike counts and check for duplicates in the same frame
            if strike_name != 'No Strike' and strike_name not in seen_strikes_this_frame:
                category = strike_categories.get(strike_name, '')
                if category:
                    strike_counts[category] += 1
                    seen_strikes_this_frame.add(strike_name)

        # Prepare overlay text with current strike counts
        display_text = ', '.join([f"{category}: {count}" for category, count in strike_counts.items() if count > 0])

        # Draw text on the current frame
        cv2.putText(frame, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 0), 2, cv2.LINE_AA)

        # Write the modified frame to the output video
        out.write(frame)
        last_frame = frame_count
        frame_count += 1  # Increment frame counter

    # Release all resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'GloryRingTestOutput.mp4'
    csv_path = 'C:/Users/12.99 a pillow/PycharmProjects/CS482FinalProject/Model Output/GloryRingTestOutput.csv'
    output_path = 'GloryRingTestAnnotatedOutput.mp4'
    overlay_strikes(video_path, csv_path, output_path)
