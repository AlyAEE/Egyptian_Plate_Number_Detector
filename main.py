import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

def plate_detection_model(video_path, model_path, device='cpu'):
    """
    Detects plates in a video using pre-trained yolo models and tracked using ByteTrack.

    Args:
        video_path (str): Path to the input video.
        model_path (str): Path to the YOLOv8 model.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Returns:
        List of (frame_number, car_id, frame_image)
    """
    model = YOLO(model_path).to(device=device)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    frame_generator = sv.get_video_frames_generator(source_path=video_path)

    frame_number = 0
    # best_results = {}
    results = []
    first_frame = next(frame_generator)
    height, width = first_frame.shape[:2]
    
    # Reset generator to first frame again
    frame_generator = sv.get_video_frames_generator(source_path=video_path)

    for frame in frame_generator:
        # Inference
        result = model.predict(frame, verbose=False)[0]

        # Filter detections by car classes
        detections = sv.Detections.from_ultralytics(result)

        # Track objects
        tracked = tracker.update_with_detections(detections)
        annotated_frame = frame.copy()
        for i in range(len(tracked)):
            x1, y1, x2, y2 = map(int, tracked.xyxy[i])
            bbox = (x1, y1, x2, y2)
            track_id = int(tracked.tracker_id[i])
            class_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1
            confidence = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # # Update best result if this confidence is higher
            # if (track_id not in best_results) or (confidence > best_results[track_id][3]):
            # best_results[track_id] = (frame_number, track_id, class_id, confidence, annotated_frame.copy(), bbox)
            results.append((frame_number, track_id, confidence, annotated_frame, bbox))
        frame_number += 1

    # return list(best_results.values()), width, height
    return results, width, height

def crop_plate_box(detections, output_size=(640, 640)):
    """
    Crops and resizes license plate regions from frames.

    Args:
        detections (list): List of tuples from plate_detection_model.
        output_size (tuple): Desired size for output cropped images.

    Returns:
        List of tuples: (frame_number, car_id, cropped_resized_plate_image)
    """
    cropped_plates = []

    for frame_num, car_id, _, frame_img, plate_bbox in detections:
        x1, y1, x2, y2 = plate_bbox

        # Ensure coordinates are within frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_img.shape[1], x2)
        y2 = min(frame_img.shape[0], y2)

        # Crop and resize
        plate_crop = frame_img[y1:y2, x1:x2]
        resized_plate = cv2.resize(plate_crop, output_size)
        cropped_plates.append((frame_num, car_id, resized_plate))

    return cropped_plates

def detect_plate_number(detections, text_model_path, device='cpu'):
    """
    Runs text detection on cropped license plates.

    Args:
        detections (list): Output from plate_detection_model.
        text_model_path (str): Path to the text/number YOLO model.
        device (str): Device for inference.

    Returns:
        List of tuples: (frame_number, car_id, detected_texts)
    """
    cropped_plates = crop_plate_box(detections)
    text_model = YOLO(text_model_path).to(device=device)
    results = []

    for frame_num, car_id, plate_img in cropped_plates:
        result = text_model.predict(plate_img, verbose=False, conf=0.25, iou=0.40)[0]
        detections = sv.Detections.from_ultralytics(result)
        texts = []
        centers = []
        boxes_labels = []
        for i in range(len(detections.xyxy)):
            
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            class_id = int(detections.class_id[i]) if detections.class_id is not None else -1
            confidence = float(detections.confidence[i]) if detections.confidence is not None else 0.0

            label = text_model.model.names[class_id] if class_id in text_model.model.names else "?"

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # texts.append((label, confidence))
            # texts.append(label)
            texts.append(label)
            centers.append((center_x, center_y))
            # boxes_labels.append((x1, label))  # sort using x1 
        results.append((frame_num, car_id, texts, centers))
        results.sort(key=lambda x: x[3])

        # # Sort by x1 (left to right)
        # boxes_labels.sort(key=lambda x: x[0])
        # plate_number = ''.join([label for _, label in boxes_labels])
        # results.append((frame_num, car_id, plate_number))

    return results

if __name__ == "__main__":
    video = "videos/madeup.mp4"
    detections, width, height = plate_detection_model(video, model_path='models/Plate_Box_Model.pt', device='cuda')
     # Run plate number detection
    text_results = detect_plate_number(detections, text_model_path='models/Plate_Text_Numbers_Model.pt', device='cuda')

    # for frame_num, car_id, texts, detections in text_results:
        # print(f"Frame {frame_num}, Car ID {car_id}, Detected: {texts}, detections: {detections}")
    for frame_num, car_id, texts in text_results:
        print(f"Frame {frame_num}, Car ID {car_id}, Detected: {texts}")

    # cropped_plates = crop_plate_box(detections)
    # for i, (frame_num, car_id, plate_img) in enumerate(cropped_plates[:10]):
    #     cv2.imwrite(f"plates/frame{frame_num}_car{car_id}.jpg", plate_img)

#     # Define VideoWriter
#     out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))

#     print("Saving video...")

#     for frame_num, car_id, confidence, annotated_frame, plate_bbox in detections:
#         print(f"Frame: {frame_num}, Car ID: {car_id}, Confidence: {confidence}, plate bbox: {plate_bbox}")
#         out.write(annotated_frame)

# out.release()
# print("Video saved as output.avi")
