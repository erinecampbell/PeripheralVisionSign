import cv2
import numpy as np

# Load video
video_path = "/Users/eecamp/Desktop/Git/PeripheralVisionOverlay/unedited_videos/20240627_883_cam01.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = "/Users/eecamp/Desktop/Git/PeripheralVisionOverlay/edited_videos/circled_20240627_883_cam01.mp4"
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_number = 0

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Define HSV range for your purple dot
lower_purple = np.array([260 // 2, 80, 80])  # Hue is 0–180 in OpenCV
upper_purple = np.array([270 // 2, 255, 255])

# Define radius in pixels for 50° visual angle (25° radius)
## horizontal_pixelsperdegree = 1600 / 103  ≈ 15.53 pixels per degree
## vertical_pixelsperdegree   = 1200 / 77   ≈ 15.58 pixels per degree
## radius_pixels = 25° * pixelsperdegree ≈ 25 * 15.5 ≈ 388 pixels
radius_px = 930 # 120 horizontal range, 60 degree radius

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Track progress
    frame_number += 1
    print(f"\rFrame: {frame_number}/{total_frames}", end="")

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the purple dot
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)

        if M["m00"] != 0:
            gaze_x = int(M["m10"] / M["m00"])
            gaze_y = int(M["m01"] / M["m00"])

            # Make a darkened copy of the frame
            dark_frame = (frame * 0.3).astype(np.uint8)

            # Create mask where gaze area is preserved
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, (gaze_x, gaze_y), radius_px, 255, -1)

            # Convert mask to 3 channels
            mask_3c = cv2.merge([mask, mask, mask])
            inverse_mask = cv2.bitwise_not(mask_3c)

            # Extract spotlight and dark regions
            spotlight = cv2.bitwise_and(frame, mask_3c)
            background = cv2.bitwise_and(dark_frame, inverse_mask)

            # Merge spotlight and dark background
            frame = cv2.add(spotlight, background)

    # Write the modified frame
    out.write(frame)

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
