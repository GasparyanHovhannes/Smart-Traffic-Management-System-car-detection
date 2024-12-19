# Vehicle Detection and Tracking

## Description
This project implements a vehicle detection and tracking system using OpenCV and Haar cascades. The system detects vehicles crossing a predefined line in a video, counts them over time intervals, and saves the results to a CSV file. Additionally, it provides an option to query vehicle counts for specific intervals.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
   - [Syntax](#syntax)

## Installation
To install this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/GasparyanHovhannes/Smart-Traffic-Management-System-car-detection.git

2. Install Package:
   ```bash
   pip install opencv-python numpy

3. Download the Haar cascade file for vehicle detection:
   https://github.com/GasparyanHovhannes/Smart-Traffic-Management-System-car-detection/blob/main/haarcascade_car.xml
4. Place the Haar cascade file in the same directory as the script.
5. Replace the video_path variable in the script with the path to your video file.

## Usage

Run the script to start vehicle detection:
```bash
python main.py
```

### Syntax
The script processes video frames to detect vehicles crossing a predefined line. It tracks and counts vehicles over specified intervals and saves the data to a CSV file. The system provides interactive querying functionality to fetch counts for specific time intervals.

Inputs:
- A video file, specified in the video_path variable.
- Pre-trained Haar cascade XML file for detecting vehicles.

Outputs:
- Live Video Feed: Displays the video with bounding boxes around detected vehicles, a red line indicating the counting area, and timestamps overlaid on the video.
- CSV File: Saves vehicle counts for specified intervals in Detected_data.csv.
- Query Functionality: Allows querying vehicle counts for specified time intervals.

Returns:
The CSV file Detected_data.csv has the following structure:
```
Interval, Car Count
2024-12-19 14:00:00 to 14:10:00, 15
2024-12-19 14:10:00 to 14:20:00, 20
...
```

Example query:
```plaintext
Enter an interval (e.g., '14:00:00 to 14:10:00') or type 'exit' to quit: 14:00:00 to 14:10:00
Car count for 14:00:00 to 14:10:00: 15
```

Note: Make sure the Haar cascade XML file and the video file are correctly set in the script before running.
