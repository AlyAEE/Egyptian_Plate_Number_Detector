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
    best_results = {}
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
            results = (frame_number, track_id, class_id, confidence, annotated_frame.copy(), bbox)
        frame_number += 1

    # return list(best_results.values()), width, height
    return results, width, height

if __name__ == "__main__":
    video = "videos/madeup.mp4"
    detections, width, height = plate_detection_model(video, model_path='models/Plate_Box_Model.pt', device='cuda')

    # Define VideoWriter
    out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))

    print("Saving video...")

    for frame_num, car_id, class_id, confidence, annotated_frame, plate_bbox in detections:
        print(f"Frame: {frame_num}, Car ID: {car_id}, Class_id: {class_id}, Confidence: {confidence}, plate bbox: {plate_bbox}")
        out.write(annotated_frame)

out.release()
print("Video saved as output.avi")

