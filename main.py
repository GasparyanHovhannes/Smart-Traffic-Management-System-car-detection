import cv2
import numpy as np
from datetime import datetime, timedelta
import csv

# Paths for pre-trained Haar cascade classifiers
vehicle_cascade_path = 'C:\\Users\\Asus\\PycharmProjects\\pythonProject1\\haarcascade_car.xml'

# Load classifiers
vehicle_cascade = cv2.CascadeClassifier(vehicle_cascade_path)

# Load the video
video_path = 'C:\\Users\\Asus\\PycharmProjects\\pythonProject1\\traf_test.mp4'
cap = cv2.VideoCapture(video_path)

# Vehicle counter
total_vehicle_count = 0
interval_vehicle_count = 0

# Dictionary to track vehicles and their crossing status and centroids
tracked_vehicles = {}

# Timer for 40-second intervals
start_time = datetime.now()
current_interval_start = start_time
interval_duration = timedelta(seconds=10)

# Dictionary to store interval data
interval_data = {}

# Frame count
frame_number = 0

while True:
    # Read frame from the video
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to load.")
        break

    # Get the original dimensions of the frame
    height, width, _ = frame.shape

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect vehicles
    vehicles = vehicle_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # Draw a red line at the bottom of the frame
    line_y = int(height * 0.2)  # Adjust the height of the red line
    cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)

    # Get the current time
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')

    # Overlay the current time and date on the video frame (top-left corner)
    cv2.putText(frame, formatted_time, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Process vehicles
    for (x, y, w, h) in vehicles:
        # Draw bounding box around the detected vehicle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Vehicle', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate the centroid of the vehicle
        centroid_x = x + w // 2
        centroid_y = y + h  # Use the bottom of the bounding box for centroids

        matched = False

        # Check if this vehicle has been tracked already
        for vehicle_id, (prev_centroid_x, prev_centroid_y, crossed, crossing_time) in tracked_vehicles.items():
            # Proximity check based on centroid position
            if abs(centroid_x - prev_centroid_x) < 50 and abs(centroid_y - prev_centroid_y) < 50:
                # If matched, update its position and check if it's crossed the line
                if not crossed:
                    # Vehicle crosses the line for the first time
                    if centroid_y >= line_y:
                        total_vehicle_count += 1
                        interval_vehicle_count += 1
                        crossing_time = current_time.strftime('%H:%M:%S')
                        tracked_vehicles[vehicle_id] = (centroid_x, centroid_y, True, crossing_time)
                        print(
                            f"Vehicle {vehicle_id} crossed the line at {crossing_time}. Total count: {total_vehicle_count}")

                matched = True
                break

        # If no match is found, create a new ID for the vehicle
        if not matched:
            vehicle_id = len(tracked_vehicles) + 1
            tracked_vehicles[vehicle_id] = (centroid_x, centroid_y, False, None)  # Vehicle hasn't crossed the line yet

    # Check if 40 seconds have passed
    if current_time - current_interval_start >= interval_duration:
        interval_end = current_time.strftime('%H:%M:%S')
        interval_start = current_interval_start.strftime('%H:%M:%S')
        interval_key = f"{interval_start} to {interval_end}"

        # Store the result in the dictionary
        interval_data[interval_key] = interval_vehicle_count

        print(f"From {interval_start} to {interval_end}, {interval_vehicle_count} cars crossed the line.")
        interval_vehicle_count = 0  # Reset the interval counter
        current_interval_start = current_time  # Reset the interval timer

    # Resize the frame for display
    aspect_ratio = width / height
    new_width = 800  # Fixed display width
    new_height = int(new_width / aspect_ratio)  # Maintain aspect ratio
    frame_resized = cv2.resize(frame, (new_width, new_height))

    # Display the frame
    cv2.imshow('Vehicle Detection', frame_resized)

    # Stop if 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # Increment frame number
    frame_number += 1

# Release video and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Save the interval data to a CSV file
output_csv_path = "Detected_data.csv"

with open(output_csv_path, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Interval", "Car Count"])  # Header row
    for key, value in interval_data.items():
        csv_writer.writerow([key, value])

print(f"Results saved to {output_csv_path}")

# Allow user to query car counts by interval
while True:
    query = input("Enter an interval (e.g., '14:00:00 to 14:10:00') or type 'exit' to quit: ")
    if query.lower() == 'exit':
        break
    if query in interval_data:
        print(f"Car count for {query}: {interval_data[query]}")
    else:
        print(f"No data found for the interval {query}.")
