# import cv2
# import numpy as np
#
# # Load YOLOv4 configuration and weights
# config_path = 'C:\\Users\\Asus\\Downloads\\yolov4.cfg'
# weights_path = 'C:\\Users\\Asus\\Downloads\\yolov4.weights'
# names_path = 'C:\\Users\\Asus\\Downloads\\coco.names'
#
# # Load YOLO model
# net = cv2.dnn.readNet(weights_path, config_path)
#
# # Load class names
# with open(names_path, 'r') as f:
#     classes = f.read().strip().split('\n')
#
# # Load the video
# video_path = 'C:\\Users\\Asus\\Downloads\\traf_test.mp4'
# cap = cv2.VideoCapture(video_path)
#
# # Initialize count dictionary for the specified classes (vehicles and pedestrians)
# target_classes = ['car', 'bus', 'truck', 'motorbike', 'bicycle', 'person']
# object_count = {cls: 0 for cls in target_classes}
#
# # List to store detected bounding boxes (for counting)
# previous_detections = []
#
#
# def is_new_detection(new_bbox, prev_bboxes, threshold=50):
#     for prev_bbox in prev_bboxes:
#         x1, y1, w1, h1 = prev_bbox
#         x2, y2, w2, h2 = new_bbox
#
#         # Calculate distance between center points of the bounding boxes
#         center1 = (x1 + w1 / 2, y1 + h1 / 2)
#         center2 = (x2 + w2 / 2, y2 + h2 / 2)
#
#         distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
#
#         # If the distance is smaller than the threshold, consider it the same object
#         if distance < threshold:
#             return False
#     return True
#
#
# while True:
#     # Read frame from the video
#     ret, frame = cap.read()
#     if not ret:
#         print("End of video or failed to load.")
#         break
#
#     height, width, channels = frame.shape
#
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#
#     layer_names = net.getLayerNames()
#     output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
#
#     # Perform detection
#     detections = net.forward(output_layers)
#
#     # Process each detection
#     boxes = []
#     confidences = []
#     class_ids = []
#
#     for out in detections:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:  # Threshold for detection
#                 # Object detected
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#
#                 # Rectangle coordinates
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
#
#     # Apply non-maxima suppression to reduce overlapping boxes
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#
#     # List to store the bounding boxes of this frame
#     current_frame_bboxes = []
#
#     # Draw the boxes on the frame
#     if len(indexes) > 0:
#         for i in indexes.flatten():
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])
#
#             # Check if the label is in the target classes (vehicles and pedestrians)
#             if label in target_classes:
#                 # Check if the new bounding box is a new detection
#                 if is_new_detection([x, y, w, h], previous_detections):
#                     object_count[label] += 1
#                     current_frame_bboxes.append([x, y, w, h])
#
#                 # Draw rectangle for the detected object
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0) if label == "person" else (0, 0, 255), 2)
#                 cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
#     # Add the current detections to the previous detections memory
#     previous_detections.extend(current_frame_bboxes)
#
#     # Display the frame
#     cv2.imshow('Object Detection', frame)
#
#     # Stop if 'Esc' key is pressed
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
#
# # Release video and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()
#
# # Print the final count of detected objects
# print("Detection summary:")
# for obj, count in object_count.items():
#     print(f"{obj}: {count}")
